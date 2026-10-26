#!/bin/bash
echo "Fetching Moby Dick from Project Gutenberg..."
curl -sSL "https://www.gutenberg.org/cache/epub/2701/pg2701.txt" -o moby_dick.txt
echo "Download complete: saved as moby_dick.txt"
