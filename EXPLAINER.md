# Dark Sky Ordinance Semantic Scorer - Detailed Explainer

## Overview

The Dark Sky Ordinance Semantic Scorer is a Python-based tool that automatically evaluates dark sky and outdoor lighting ordinances against a comprehensive set of best-practice criteria. Using semantic similarity analysis powered by SentenceTransformer models, the tool scores how well an ordinance document addresses key dark sky protection requirements.

### What It Does

1. **Parses PDF Documents**: Extracts text from ordinance PDF files
2. **Semantic Analysis**: Uses machine learning embeddings to understand the meaning of text
3. **Criteria Matching**: Compares ordinance content against 29 dark sky criteria
4. **Scoring**: Provides per-criterion scores (0-100) and an overall weighted score
5. **Evidence Extraction**: Shows the most relevant excerpts from the ordinance for each criterion

### Key Features

- **Fully Local**: No external API calls - all processing happens on your machine
- **Semantic Understanding**: Uses advanced NLP to understand context, not just keyword matching
- **Comprehensive Criteria**: 29 criteria covering all aspects of dark sky ordinances
- **Weighted Scoring**: Important criteria (like shielding requirements) have higher weights
- **Evidence-Based**: Shows exactly which parts of the ordinance match each criterion

---

## How It Works

### Technical Architecture

```
PDF Document
    ↓
[PDF Parser] → Extracts raw text
    ↓
[Text Splitter] → Chunks text into overlapping segments
    ↓
[SentenceTransformer] → Converts text to numerical embeddings
    ↓
[Semantic Scorer] → Compares document chunks to criteria embeddings
    ↓
[Results] → Scores + Evidence excerpts
```

### Step-by-Step Process

#### 1. **PDF Text Extraction**
- Uses `pdfplumber` library to extract text from all pages
- Handles multi-page documents
- Preserves basic text structure

#### 2. **Text Chunking**
- Splits document into overlapping chunks (default: 2000 characters with 200 character overlap)
- Overlap ensures context isn't lost at chunk boundaries
- Each chunk is treated as a separate unit for analysis

**Why chunking?**
- Large documents can't be processed as single units
- Chunking allows finding specific relevant sections
- Overlap ensures sentences aren't split awkwardly

#### 3. **Semantic Embedding**
- Uses SentenceTransformer models (default: `all-MiniLM-L6-v2`)
- Converts text chunks and criteria descriptions into high-dimensional vectors (embeddings)
- Embeddings capture semantic meaning, not just words

**Example:**
- "Lights must be shielded" and "Fixtures shall prevent light from escaping upward" 
- These have similar meanings and will have similar embeddings, even with different wording

#### 4. **Similarity Calculation**
- For each criterion, computes cosine similarity with all document chunks
- Cosine similarity measures the cosine of the angle between two vectors: **1 = identical direction** (most similar), **0 = orthogonal** (unrelated), **−1 = opposite direction** (most dissimilar)
- Finds the highest similarity score for each criterion

#### 5. **Score Normalization**
- Converts similarity scores (typically 0.3-0.9) to 0-100 scale
- Formula: `score = max(similarity, 0) * 100`
- Negative similarities (dissimilar content) are set to 0

#### 6. **Weighted Overall Score**
- Each criterion has a weight (1.0 to 1.8)
- More important criteria (e.g., shielding requirements) have higher weights
- Overall score = weighted average of all criterion scores

---

## Scoring Methodology

### Per-Criterion Scoring

Each of the 29 criteria is scored independently:

1. **Embedding Generation**: Criterion description is converted to an embedding vector
2. **Document Comparison**: Every document chunk is compared to the criterion embedding
3. **Best Match**: The highest similarity score is taken as the criterion score
4. **Evidence Extraction**: The top-k matching chunks are saved as evidence

### Score Interpretation

- **90-100**: Excellent - Ordinance strongly addresses this criterion
- **70-89**: Good - Ordinance adequately addresses this criterion
- **50-69**: Fair - Ordinance partially addresses this criterion
- **30-49**: Poor - Ordinance minimally addresses this criterion
- **0-29**: Very Poor - Ordinance does not address this criterion

### Overall Score

The overall score is a weighted average:
- More critical criteria (shielding, light trespass) have weights of 1.7-1.8
- Important criteria (color temperature, extinguish times) have weights of 1.3-1.6
- Supporting criteria (exemptions, holiday lighting) have weights of 0.9-1.2

**Example Calculation:**
```
Criterion 1 (weight 1.8): Score 85 → Contribution: 85 × 1.8 = 153
Criterion 2 (weight 1.5): Score 70 → Contribution: 70 × 1.5 = 105
Criterion 3 (weight 1.0): Score 60 → Contribution: 60 × 1.0 = 60
Total weight: 1.8 + 1.5 + 1.0 = 4.3
Overall score: (153 + 105 + 60) / 4.3 = 74.0
```

---

## The 29 Dark Sky Criteria

### Core Requirements (High Weight: 1.5-1.8)

1. **Purpose Statement** (1.5) - Clear statement protecting dark skies
2. **Shielding Requirements** (1.8) - Lights must be downcast and fully shielded
3. **Light Trespass Standard** (1.7) - Prohibition of visible light from off-site
4. **Color Temperature (CCT)** (1.6) - Limits on color temperature (amber/yellow preferred)
5. **Uplighting Prohibition** (1.5) - No upward-directed lighting
6. **Lighting Classifications** (1.5) - Class 1, 2, 3 lighting definitions

### Operational Requirements (Medium-High Weight: 1.3-1.4)

7. **Commercial Extinguish Times** (1.4) - Lights off by 11 PM or after business hours
8. **Lumen Output Limits - Non-Residential** (1.4) - Maximum lumens per acre
9. **Public Street Lights** (1.4) - BUG rating requirements
10. **Parking Lot Lighting** (1.4) - Shielded amber lights, off times
11. **Lighting Plans and Approvals** (1.4) - Required for new construction

### Specific Applications (Medium Weight: 1.1-1.3)

12. **Applicability and Scope** (1.3) - Defines what outdoor lighting is covered
13. **Residential Extinguish Times** (1.3)
14. **Lumen Output Limits - Residential** (1.3)
15. **Security Lights** (1.3)
16. **Sports and Recreational Lighting** (1.3)
17. **Enforcement and Penalties** (1.3)
18. **Motion Sensors and Smart Lighting** (1.2)
19. **Decorative and Architectural Lighting** (1.2)
20. **Inspection Requirements** (1.2)
21. **Greenhouse Lighting** (1.2)
22. **Outdoor Display Lots** (1.2)

### Supporting Elements (Lower Weight: 0.9-1.1)

23. **Internally Lighted Signs** (1.1)
24. **Compliance Timeline and Grandfathering** (1.1)
25. **Externally Lighted Signs** (1.1)
26. **Digital Billboard Signs** (1.1)
27. **String Lights** (1.0)
28. **Exemptions** (1.0)
29. **Holiday Decorative Lighting** (0.9)

---

## Usage Guide

### Streamlit Web Interface

**Start the app:**
```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

**Using the interface:**

1. **Upload PDF**: Click "Upload ordinance PDF" and select your ordinance file
2. **Adjust Settings** (optional):
   - **Model**: Choose SentenceTransformer model (default: all-MiniLM-L6-v2)
   - **Chunk Size**: Text chunk size in characters (default: 2000)
   - **Chunk Overlap**: Overlap between chunks (default: 200)
   - **Top Excerpts**: Number of evidence excerpts per criterion (default: 1)
3. **Run Analysis**: Click "Run semantic scoring"
4. **Review Results**:
   - Overall score at the top
   - Per-criterion scores with evidence excerpts
   - Download JSON report button

**What to look for:**
- Low scores indicate missing or weak provisions
- Evidence excerpts show where the ordinance addresses (or doesn't address) each criterion
- Use evidence to identify specific sections that need improvement

### Running the Test Suite

All tests live in `tests/` and require no internet access — the ML model and PDF parser are mocked.

**Run all tests:**
```bash
pytest
```

**Unit tests only** (fast, no heavy dependencies):
```bash
pytest tests/unit/
```

**Integration tests only** (wires the full pipeline with mocked embeddings):
```bash
pytest tests/integration/
```

**Useful flags:**
```bash
pytest -x          # stop on the first failure
pytest -s          # show print() output while running
pytest -k scorer   # run only tests whose name contains "scorer"
```

**Test layout:**
```
tests/
├── conftest.py                  # shared fixtures (mock model, mock PDF, criteria)
├── unit/
│   ├── test_text_splitter.py    # chunk_text() behaviour and edge cases
│   ├── test_pdf_parser.py       # extract_text_from_pdf() with mocked pdfplumber
│   ├── test_embeddings.py       # cosine_similarity() and EmbeddingProvider
│   └── test_scorer.py           # OrdinanceScorer scoring logic and weight math
└── integration/
    └── test_pipeline.py         # end-to-end pipeline + criteria.json integrity
```

---

### Command-Line Interface

**Score a single PDF:**
```bash
python scripts/score_samples.py \
  --sample-dir sample \
  --criteria app/data/criteria.json \
  --model all-MiniLM-L6-v2 \
  --chunk-size 2000 \
  --chunk-overlap 200 \
  --top-k 1 \
  --out-dir out
```

**Parameters:**
- `--sample-dir`: Directory containing PDF files to score
- `--criteria`: Path to criteria JSON file
- `--model`: SentenceTransformer model name
- `--chunk-size`: Size of text chunks in characters
- `--chunk-overlap`: Overlap between chunks
- `--top-k`: Number of top excerpts to save per criterion
- `--out-dir`: Directory for output JSON reports

**Output:**
- Creates JSON files in `out/` directory
- Each file named `{pdf_name}_score.json`
- Contains overall score, per-criterion scores, and evidence excerpts

---

## Understanding the Results

### JSON Report Structure

```json
{
  "meta": {
    "file": "path/to/ordinance.pdf",
    "model": "all-MiniLM-L6-v2",
    "num_chunks": 45,
    "overall_score": 72.5
  },
  "criteria_results": [
    {
      "title": "Shielding Requirements",
      "short": "Are lights required to be downcast and fully shielded?",
      "score": 85.3,
      "raw_similarity": 0.853,
      "top_excerpts": [
        "All outdoor lighting fixtures shall be fully shielded and downcast..."
      ],
      "top_scores": [0.853],
      "weight": 1.8
    },
    ...
  ]
}
```

### Interpreting Scores

**High Score (80+)**: 
- Ordinance clearly addresses the criterion
- Evidence excerpts show strong language
- May need minor refinement

**Medium Score (50-79)**:
- Ordinance partially addresses the criterion
- May be vague or incomplete
- Needs strengthening

**Low Score (<50)**:
- Criterion is missing or very weak
- Major revision needed
- Consider adding new sections

### Using Evidence Excerpts

Evidence excerpts show the actual text from your ordinance that matches each criterion. Use them to:
- Verify the scoring is accurate
- Identify where improvements are needed
- Find specific sections to revise
- Understand why a score is high or low

---

## Model Selection

### Default Model: `all-MiniLM-L6-v2`

**Pros:**
- Fast (small model, ~80MB)
- Good balance of speed and accuracy
- Works well for general text similarity

**Cons:**
- May miss nuanced technical language
- Less accurate for domain-specific terms

### Alternative Models

**For Better Accuracy:**
- `all-mpnet-base-v2` - Larger, more accurate (420MB)
- `paraphrase-multilingual-mpnet-base-v2` - Multilingual support

**For Faster Processing:**
- `all-MiniLM-L12-v2` - Slightly larger, slightly better
- `all-distilroberta-v1` - Distilled model, faster

**Changing the model:**
- In Streamlit: Use the sidebar input
- In CLI: Use `--model` parameter
- First run downloads the model (may take time)

---

## Best Practices

### For Ordinance Authors

1. **Be Specific**: Use clear, detailed language rather than vague statements
2. **Use Technical Terms**: Include proper terminology (e.g., "full cutoff", "BUG rating")
3. **Provide Examples**: Include examples of compliant and non-compliant lighting
4. **Define Terms**: Have a definitions section for technical terms
5. **Be Comprehensive**: Address all 29 criteria for best scores

### For Evaluators

1. **Review Evidence**: Always check the evidence excerpts, not just scores
2. **Context Matters**: Low scores may indicate missing sections, not poor quality
3. **Compare Multiple Ordinances**: Use scores to compare different ordinances
4. **Iterate**: Use results to identify areas for improvement
5. **Manual Review**: Use tool as a guide, but always do manual review

### For Developers

1. **Customize Criteria**: Edit `app/data/criteria.json` to add/modify criteria
2. **Adjust Weights**: Change weights based on local priorities
3. **Experiment with Models**: Try different SentenceTransformer models
4. **Tune Chunking**: Adjust chunk size/overlap for your document types
5. **Extend Functionality**: Add new features (e.g., comparison mode, trend analysis)

---

## Limitations and Considerations

### What the Tool Does Well

✅ Identifies if criteria are addressed in the ordinance
✅ Finds relevant sections automatically
✅ Provides quantitative scores for comparison
✅ Handles various writing styles and formats
✅ Works with any PDF ordinance document

### What the Tool Cannot Do

❌ Evaluate legal quality or enforceability
❌ Check for contradictions or conflicts
❌ Verify compliance with local laws
❌ Assess political feasibility
❌ Replace expert legal review

### Important Notes

- **Semantic similarity is not perfect**: The tool may miss some matches or find false positives
- **Context matters**: A low score doesn't always mean the ordinance is bad
- **Language variations**: Different wording for the same concept may score differently
- **Model limitations**: Smaller models may miss technical nuances
- **PDF quality**: Poorly scanned PDFs may have extraction errors

---

## Troubleshooting

### Common Issues

**"No text found in PDF"**
- PDF may be image-based (scanned)
- Try OCR software first
- Check if PDF is corrupted

**Low scores across the board**
- Ordinance may use different terminology
- Try a larger model (e.g., `all-mpnet-base-v2`)
- Check if text extraction worked correctly

**Model download fails**
- Check internet connection
- Try downloading model manually
- Check disk space (models can be 100-500MB)

**Import errors**
- Make sure virtual environment is activated
- Install requirements: `pip install -r requirements.txt`
- Check Python version (3.9+)

**Slow processing**
- Use smaller model (`all-MiniLM-L6-v2`)
- Reduce chunk size
- Process fewer documents at once

---

## Advanced Usage

### Customizing Criteria

Edit `app/data/criteria.json` to:
- Add new criteria
- Modify descriptions
- Adjust weights
- Change titles/short descriptions

**Example:**
```json
{
  "title": "Custom Criterion",
  "short": "Short description",
  "description": "Full detailed description of what to look for...",
  "weight": 1.5
}
```

### Batch Processing

Process multiple ordinances:
```bash
# Put all PDFs in a directory
python scripts/score_samples.py --sample-dir ordinances/ --out-dir results/
```

### Comparing Ordinances

1. Score multiple ordinances
2. Compare JSON outputs
3. Look for patterns in low-scoring criteria
4. Identify best practices from high-scoring ordinances

### Integration with Other Tools

The JSON output can be:
- Imported into spreadsheets for analysis
- Used in data visualization tools
- Integrated into ordinance management systems
- Processed by other analysis scripts

---

## Technical Details

### Dependencies

- **streamlit**: Web interface framework
- **pdfplumber**: PDF text extraction
- **sentence-transformers**: Semantic embeddings
- **numpy**: Numerical computations
- **scikit-learn**: (dependency of sentence-transformers)

### Performance

- **Processing time**: ~10-30 seconds per document (depending on size and model)
- **Memory usage**: ~500MB-2GB (depending on model)
- **Model download**: One-time, ~80-500MB depending on model

### File Structure

```
ordinance-semantic-scorer/
├── app/
│   ├── analysis/
│   │   ├── embeddings.py      # SentenceTransformer wrapper
│   │   └── scorer.py           # Scoring logic
│   ├── data/
│   │   └── criteria.json       # 29 dark sky criteria
│   └── utils/
│       ├── pdf_parser.py        # PDF extraction
│       └── text_splitter.py     # Text chunking
├── scripts/
│   └── score_samples.py         # CLI interface
└── streamlit_app.py            # Web interface
```

---

## Future Enhancements

Potential improvements:
- Multi-language support
- Comparison mode (side-by-side ordinance comparison)
- Trend analysis (tracking ordinance improvements over time)
- Custom criterion templates
- Export to Word/PDF reports
- Integration with ordinance databases
- Confidence scores for matches
- Highlighting of relevant sections in PDF

---

## Support and Contributing

### Getting Help

- Check this explainer document
- Review the README.md
- Examine example outputs
- Test with known-good ordinances

### Contributing

- Report issues with scoring accuracy
- Suggest new criteria
- Improve documentation
- Add new features

---

## Conclusion

The Dark Sky Ordinance Semantic Scorer is a powerful tool for evaluating dark sky ordinances. It provides objective, quantitative scores based on semantic analysis, helping ordinance authors and evaluators identify strengths and weaknesses. While it's not a replacement for expert review, it serves as an excellent starting point and comparison tool.

Remember: The tool analyzes what's *in* the ordinance, not how well it's written or how enforceable it is. Always combine automated scoring with expert legal and policy review.
