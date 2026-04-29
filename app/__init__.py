"""
Dark Sky Ordinance Semantic Scorer — application package.

This package scores municipal or organizational **outdoor lighting ordinances**
against a fixed rubric of dark-sky-friendly criteria (see ``app/criteria.json``).

**Submodules**

* ``app.scorer`` — ``OrdinanceScorer``: embed text with Sentence-Transformers and
  compute weighted semantic similarity scores.
* ``app.utils`` — PDF text extraction (``pdfplumber``) and character-based chunking
  for long documents.

**Typical entry points**

* Run the Streamlit UI from the project root: ``streamlit run streamlit_app.py``.
* Import ``OrdinanceScorer`` and ``chunk_text`` / ``extract_text_from_pdf`` from
  application code or tests.
"""
