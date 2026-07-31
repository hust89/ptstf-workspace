use ptstf_core::graph::TrigramGraph;
use ptstf_core::tensor::{TransitionTensor, NcpFactors};
use ptstf_core::flow::TriadicFlowCurve;
use ptstf_core::optimization::{GradientFlowOptimizer, smooth_crossing_probability};
use nalgebra::{DMatrix, Vector2, SymmetricEigen};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Write, Read};
use std::time::Instant;
use std::env;

// Import parallel iteration traits explicitly
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[derive(Serialize)]
struct OutputReport {
    vocabulary: Vec<String>,
    embeddings: Vec<(f64, f64)>,
    topic_centroids: Vec<(f64, f64)>,
    total_trigrams_processed: usize,
    pipeline_duration_seconds: f64,
}

fn main() {
    println!("============================================================");
    println!("📡 PTSTF Version 1.1 - Industrial NLP & Spectral Flow Pipeline");
    println!("============================================================\n");

    // Parse Command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("❌ Error: Missing input corpus path.");
        println!("Usage: cargo run --release -p ptstf-core --bin ptstf_pipeline <path_to_text_file>\n");
        println!("💡 Tip: You can download a free public-domain book to use as a corpus via:");
        println!("   curl -s https://www.gutenberg.org/files/11/11-0.txt -o alice.txt");
        println!("   cargo run --release -p ptstf-core --bin ptstf_pipeline alice.txt\n");
        return;
    }
    
    let corpus_path = &args[1];
    println!("📖 Loading raw text corpus from: '{}'...", corpus_path);

    let start_pipeline = Instant::now();

    // ==========================================
    // 1. Text Parsing & Tokenization
    // ==========================================
    let mut file_content = String::new();
    if let Err(e) = File::open(corpus_path).and_then(|mut f| f.read_to_string(&mut file_content)) {
        println!("❌ Error reading corpus file: {}", e);
        return;
    }

    println!("   ✔ Loaded {} characters of text.", file_content.len());
    
    // Corrected token map evaluation
    let words: Vec<String> = file_content
        .split_whitespace()
        .map(|w| {
            let mut s = w.to_lowercase();
            s.retain(|c: char| c.is_alphanumeric() || c == '\'');
            s
        })
        .filter(|w| !w.is_empty())
        .collect();

    println!("   ✔ Parsed {} clean tokens.", words.len());

    // Compute Word Frequencies & Select Top Vocabulary
    let mut unigram_counts = HashMap::new();
    for word in &words {
        *unigram_counts.entry(word.clone()).or_insert(0) += 1;
    }

    let mut vocab_freqs: Vec<(String, usize)> = unigram_counts.into_iter().collect();
    vocab_freqs.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Select Top M frequent words (industry setting: scale vocabulary dimension)
    let vocab_size = 120; // 120-word vocabulary
    let vocabulary: Vec<String> = vocab_freqs.into_iter()
        .take(vocab_size)
        .map(|(w, _)| w)
        .collect();

    let vocab_map: HashMap<String, usize> = vocabulary.iter().cloned().enumerate()
        .map(|(idx, w)| (w, idx))
        .collect();

    println!("   ✔ Generated Vocabulary of top {} semantic hubs.", vocab_size);

    // ==========================================
    // 2. Bigram Co-occurrence & Graph Laplacian
    // ==========================================
    println!("\n2. Extracting Word Co-occurrence Graph & Normalized Laplacian...");
    let mut adjacency = DMatrix::from_element(vocab_size, vocab_size, 0.0);
    for window in words.windows(2) {
        if let (Some(&i), Some(&j)) = (vocab_map.get(&window[0]), vocab_map.get(&window[1])) {
            adjacency[(i, j)] += 1.0;
            adjacency[(j, i)] += 1.0; // Symmetric undirected trigram overlap framework
        }
    }

    let graph = TrigramGraph::new(adjacency).unwrap();
    let laplacian = graph.compute_laplacian();

    // Perform REAL Spectral Eigen-decomposition
    println!("   ✔ Performing Symmetric Eigen-Decomposition of Laplacian...");
    let eigen = SymmetricEigen::new(laplacian);
    let mut eigen_pairs: Vec<(f64, usize)> = eigen.eigenvalues.iter().cloned().enumerate()
        .map(|(idx, val)| (val, idx))
        .collect();
    
    // Sort eigenvalues to extract the 2nd and 3rd smallest coordinates (Spectral Chart Map)
    eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let fiedler_idx = eigen_pairs[1].1;
    let third_ev_idx = eigen_pairs[2].1;

    let fiedler_vector = eigen.eigenvectors.column(fiedler_idx);
    let third_vector = eigen.eigenvectors.column(third_ev_idx);

    let mut embeddings = vec![Vector2::new(0.0, 0.0); vocab_size];
    for i in 0..vocab_size {
        embeddings[i] = Vector2::new(fiedler_vector[i], third_vector[i]);
    }
    println!("   ✔ Word representations mapped into Spectral Space R^2.");

    // ==========================================
    // 3. Conditional Probability Tensor & NCP
    // ==========================================
    println!("\n3. Building 3D Conditional Transition Probability Tensor...");
    let mut transition_counts = vec![0.0; vocab_size * vocab_size * vocab_size];
    let mut bigram_counts = HashMap::new();

    let mut trigram_total = 0;
    for window in words.windows(3) {
        if let (Some(&i), Some(&j), Some(&k)) = (vocab_map.get(&window[0]), vocab_map.get(&window[1]), vocab_map.get(&window[2])) {
            transition_counts[i + j * vocab_size + k * vocab_size * vocab_size] += 1.0;
            *bigram_counts.entry((i, j)).or_insert(0.0) += 1.0;
            trigram_total += 1;
        }
    }

    // Convert raw counts into conditional probabilities: P(w_k | w_i, w_j)
    for i in 0..vocab_size {
        for j in 0..vocab_size {
            let total = bigram_counts.get(&(i, j)).cloned().unwrap_or(0.0);
            if total > 0.0 {
                for k in 0..vocab_size {
                    transition_counts[i + j * vocab_size + k * vocab_size * vocab_size] /= total;
                }
            }
        }
    }

    let tensor = TransitionTensor::new(transition_counts, vocab_size).unwrap();
    println!("   ✔ Generated Transition Probability Tensor ({} entries).", vocab_size.pow(3));
    println!("   ✔ Total trigram occurrences processed: {}.", trigram_total);

    println!("   ✔ Performing Non-Negative Canonical Polyadic Decomposition (Rank 5)...");
    let factors = tensor.decompose(5, 10).unwrap(); // Rank 5 topic modeling, 10 iterations

    // Compute Topic Barycenter Centroids
    let mut centroids = Vec::new();
    for r in 0..5 {
        if let Ok(centroid) = factors.compute_stable_centroid(1, r, &embeddings) {
            centroids.push(centroid);
        }
    }
    println!("   ✔ Generated {} stable, division-by-zero free semantic centroids.", centroids.len());

    // ==========================================
    // 4. Parallel Curve Auditing (Rayon)
    // ==========================================
    println!("\n4. Spawning Triadic Transition Flow Pathways...");
    let mut curves = Vec::new();
    let mut curve_indices = Vec::new();
    let mut alphas = Vec::new();

    // Generate transition curves for each non-zero transition
    for window in words.windows(3) {
        if let (Some(&i), Some(&j), Some(&k)) = (vocab_map.get(&window[0]), vocab_map.get(&window[1]), vocab_map.get(&window[2])) {
            curves.push(TriadicFlowCurve::new(embeddings[i], embeddings[j], embeddings[k]));
            curve_indices.push((i, j, k));
            alphas.push(1.0);
        }
    }

    println!("   ⚡ Auditing {} transition pathways in parallel via Rayon...", curves.len());
    let start_audit = Instant::now();
    let energies: Vec<f64> = curves.par_iter().map(|c| c.bending_energy(1.0, 1.0)).collect();
    let crossings: Vec<bool> = curves.par_iter().map(|c| c.detect_crossing()).collect();
    let audit_duration = start_audit.elapsed();

    println!("   ✔ Parallel auditing complete in {:.2?}.", audit_duration);
    println!("   ✔ Throughput: {:.2} pathways/second.", (curves.len() as f64) / audit_duration.as_secs_f64());

    // ==========================================
    // 5. Optimization & Representation Flow
    // ==========================================
    println!("\n5. Running Representation Gradient Flow Optimization (Theorem 9.1)...");
    let optimizer = GradientFlowOptimizer::new(0.00000001, 10.0, 1.5);
    let steps = [0.0, 0.25, 0.5, 0.75, 1.0];

    // Compute original Corpus Objective
    let eval_objective = |coords: &[Vector2<f64>]| -> f64 {
        curve_indices.iter().enumerate().map(|(idx, &indices)| {
            let curve = TriadicFlowCurve::new(coords[indices.0], coords[indices.1], coords[indices.2]);
            let energy = curve.bending_energy(1.0, 1.0);
            let p_cross = smooth_crossing_probability(&curve, &steps, optimizer.tau);
            - optimizer.mu * alphas[idx] * energy * p_cross
        }).sum()
    };

    let initial_obj = eval_objective(&embeddings);
    println!("   ✔ Epoch 0 Initial Corpus Objective J(Y): {:.6}", initial_obj);

    let mut last_obj = initial_obj;
    for epoch in 1..=3 {
        optimizer.step_representation_flow(&mut embeddings, &curve_indices, &alphas, 1.0, 1.0);
        let current_obj = eval_objective(&embeddings);
        println!("   ✔ Epoch {} Corpus Objective J(Y): {:.6} (Diff: {:+.6})", 
            epoch, current_obj, current_obj - last_obj);
        
        assert!(current_obj >= last_obj, "Strict Monotonicity Violation!");
        last_obj = current_obj;
    }

    println!("   ✔ Theorem 9.1 Verified: Continuous Monotonic Ascent Confirmed.");

    // ==========================================
    // 6. Output Metadata Serialization
    // ==========================================
    println!("\n6. Exporting Industrial Metadata Reports...");
    let report = OutputReport {
        vocabulary: vocabulary.clone(),
        embeddings: embeddings.iter().map(|e| (e.x, e.y)).collect(),
        topic_centroids: centroids.iter().map(|c| (c.x, c.y)).collect(),
        total_trigrams_processed: trigram_total,
        pipeline_duration_seconds: start_pipeline.elapsed().as_secs_f64(),
    };

    let report_file = "ptstf_analysis_report.json";
    let report_content = serde_json::to_string_pretty(&report).unwrap();
    let mut file = File::create(report_file).unwrap();
    file.write_all(report_content.as_bytes()).unwrap();

    println!("   ✔ Saved comprehensive semantic report to: '{}'\n", report_file);

    println!("============================================================");
    println!("🎉 PTSTF Pipeline Finished in {:.2?}! System Stable.", start_pipeline.elapsed());
    println!("============================================================");
}
