use tonic::{transport::Server, Request, Response, Status};
use ptstf::ptstf_service_server::{PtstfService, PtstfServiceServer};
use ptstf::{ConductanceRequest, ConductanceResponse, FlowRequest, FlowResponse, AuditResult};

use ptstf_core::embedding::ConductanceField;
use ptstf_core::flow::TriadicFlowCurve;
use nalgebra::Vector2;

pub mod ptstf {
    tonic::include_proto!("ptstf");
}

#[derive(Debug, Default)]
pub struct MyPtstfService;

#[tonic::async_trait]
impl PtstfService for MyPtstfService {
    async fn evaluate_conductance(
        &self,
        request: Request<ConductanceRequest>,
    ) -> Result<Response<ConductanceResponse>, Status> {
        let req = request.into_inner();
        let target = req.target.ok_or_else(|| Status::invalid_argument("Missing target"))?;
        
        let embeddings: Vec<Vector2<f64>> = req.embeddings.into_iter()
            .map(|c| Vector2::new(c.x, c.y))
            .collect();

        let field = ConductanceField::new(embeddings, req.discrete_conductances, req.eta)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let value = field.evaluate(Vector2::new(target.x, target.y));

        Ok(Response::new(ConductanceResponse { value }))
    }

    async fn audit_flow_curves(
        &self,
        request: Request<FlowRequest>,
    ) -> Result<Response<FlowResponse>, Status> {
        let req = request.into_inner();
        
        let results = req.curves.into_iter()
            .map(|c| {
                let p0 = c.p0.unwrap_or_default();
                let p1 = c.p1.unwrap_or_default();
                let p2 = c.p2.unwrap_or_default();
                
                let curve = TriadicFlowCurve::new(
                    Vector2::new(p0.x, p0.y),
                    Vector2::new(p1.x, p1.y),
                    Vector2::new(p2.x, p2.y),
                );

                AuditResult {
                    bending_energy: curve.bending_energy(req.a, req.b),
                    crosses_boundary: curve.detect_crossing(),
                }
            })
            .collect();

        Ok(Response::new(FlowResponse { results }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let service = MyPtstfService;

    println!("⚡ PTSTF Microservice Server running on {}", addr);

    Server::builder()
        .add_service(PtstfServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
