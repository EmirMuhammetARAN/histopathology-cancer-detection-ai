 # Contributing to Medical Cancer Detection Project

Thank you for your interest in contributing! This is a **research project** focused on advancing medical AI responsibly.

## 🎯 Contribution Areas

### 1. External Validation (High Priority)
- Test model on independent datasets (Camelyon17, TCGA, etc.)
- Report performance metrics with confidence intervals
- Document dataset characteristics and limitations

### 2. Clinical Translation
- Whole-slide image integration
- Real-world workflow integration (PACS/LIS)
- Prospective study design proposals

### 3. Technical Improvements
- Cross-validation implementation
- Uncertainty quantification (Bayesian, ensembles)
- Model optimization (pruning, quantization)
- Real-time inference (<10ms per patch)

### 4. Interpretability & Bias
- Enhanced Grad-CAM visualizations
- Subgroup analysis (demographics, cancer stage)
- Bias detection and mitigation
- Failure mode analysis

### 5. Documentation
- Tutorial notebooks
- Deployment guides
- Clinical use case studies
- Regulatory pathway documentation

## 📋 Before Contributing

### Prerequisites
1. **Understand Medical Context:** This is healthcare AI - patient safety is paramount
2. **Read Documentation:** Review README, VALIDATION_REPORT, LICENSE
3. **Ethics Training:** Understand HIPAA, GDPR, medical device regulations
4. **IRB Approval:** If using patient data, obtain institutional review

### Code of Conduct
- ✅ Patient safety first - always
- ✅ Honest reporting (no cherry-picking results)
- ✅ Acknowledge limitations transparently
- ✅ Respect data privacy (no PHI in issues/PRs)
- ❌ No overhyping performance ("AI outperforms doctors")
- ❌ No unauthorized clinical use

## 🔄 Development Workflow

### 1. Fork & Clone
```bash
git clone https://github.com/yourusername/patches.git
cd patches
git checkout -b feature/your-feature-name
```

### 2. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install pytest pytest-cov  # For testing
```

### 3. Development Guidelines

**Code Style:**
- Follow PEP 8 (Python)
- Use type hints where possible
- Document functions with docstrings (Google style)
- Comment complex medical logic

**Testing:**
- Add unit tests for new features (`tests/test_*.py`)
- Ensure existing tests pass: `pytest tests/ -v`
- Aim for >80% code coverage

**Reproducibility:**
- Always set random seeds (42)
- Document software versions
- Include environment files (requirements.txt)

### 4. Medical AI Best Practices

**Data Handling:**
```python
# ✅ Good: Anonymize data
patient_id = hash_patient_id(original_id)

# ❌ Bad: Expose PHI
patient_name = "John Doe"  # HIPAA violation!
```

**Performance Reporting:**
```python
# ✅ Good: Report confidence intervals
print(f"Sensitivity: {recall:.1%} (95% CI: [{ci_low:.1%}, {ci_high:.1%}])")

# ❌ Bad: Single point estimate
print(f"Sensitivity: {recall:.1%}")  # Misleading!
```

**Limitations:**
```markdown
✅ Good: "This model was tested on PCam only and may not generalize."
❌ Bad: "This model works on all cancer types." (Overgeneralization)
```

### 5. Commit Messages
```
feat: Add cross-validation with 5-fold stratified splits
fix: Correct AUC calculation for imbalanced datasets
docs: Update README with external validation results
test: Add unit tests for Grad-CAM heatmap generation
```

### 6. Pull Request Process

**Before Submitting:**
- [ ] Code passes all tests (`pytest`)
- [ ] Documentation updated (README, docstrings)
- [ ] No PHI or sensitive data in commits
- [ ] Reproducibility maintained (seeds, versions)
- [ ] Medical limitations acknowledged

**PR Description Template:**
```markdown
## Changes
Brief description of what you changed.

## Motivation
Why was this change necessary?

## Testing
How did you test this? Include commands/results.

## Clinical Impact
Does this affect model predictions? Patient safety considerations?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No PHI exposed
- [ ] Limitations acknowledged
```

### 7. Review Process
1. Automated tests run (GitHub Actions)
2. Code review by maintainer
3. Medical safety review (if affects predictions)
4. Documentation review
5. Merge after approval

## 🚫 What NOT to Contribute

❌ **Patient Data:** Never commit real patient images/data  
❌ **PHI:** No names, IDs, dates, locations  
❌ **Proprietary Code:** Only open-source compatible  
❌ **Untested Code:** Must include tests  
❌ **Clinical Claims:** No "FDA approved" or "replaces pathologists"  
❌ **Cherry-Picked Results:** Report all experiments, not just best  

## 📊 External Validation Contributions

If you're testing on a new dataset, please include:

### Minimum Requirements
- Dataset description (source, size, demographics)
- Preprocessing details (must match training)
- Performance metrics (AUC, sensitivity, precision with 95% CI)
- Confusion matrix and error analysis
- Limitations and biases identified

### Example Report Structure
```markdown
## External Validation: [Dataset Name]

**Dataset:** [Source, year, institution]  
**Samples:** N=X,XXX images  
**Tissue Type:** [lymph node/breast/etc.]  
**Scanner:** [vendor, model]  

**Results:**
| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC-ROC | 0.XX | [0.XX, 0.XX] |
| Sensitivity | XX% | [XX%, XX%] |
| Precision | XX% | [XX%, XX%] |

**Limitations:**
- [List any issues: image quality, label noise, etc.]

**Conclusion:**
[Does model generalize? Where does it fail?]
```

## 🧪 Adding Tests

```python
# tests/test_new_feature.py

import pytest
import numpy as np

def test_new_function():
    """Test description following medical context."""
    # Setup
    input_data = ...
    expected_output = ...
    
    # Execute
    result = new_function(input_data)
    
    # Assert
    assert result == expected_output
    
    # Medical safety check
    assert 0.0 <= result <= 1.0, "Probability must be in [0,1]"
```

## 📝 Documentation Standards

### Docstrings (Google Style)
```python
def predict_cancer(image: np.ndarray, threshold: float = 0.40) -> dict:
    """
    Predict cancer probability from histopathology image.
    
    ⚠️ WARNING: Research use only. Not for clinical diagnosis.
    
    Args:
        image: Preprocessed image array (50, 50, 3) normalized
        threshold: Decision threshold optimized for high sensitivity
    
    Returns:
        Dictionary containing:
            - probability: Cancer probability [0.0, 1.0]
            - prediction: Binary class (0=benign, 1=cancer)
            - confidence: Prediction confidence score
    
    Raises:
        ValueError: If image shape is incorrect
        
    Example:
        >>> img = preprocess_image("sample.png")
        >>> result = predict_cancer(img)
        >>> print(f"Cancer prob: {result['probability']:.1%}")
    
    Clinical Note:
        High sensitivity (>90%) optimized to minimize false negatives.
        May produce false positives requiring pathologist review.
    """
    pass
```

## 🔒 Data Privacy Requirements

**Before Using Patient Data:**
1. ✅ Obtain IRB approval
2. ✅ Get informed consent (or waiver)
3. ✅ Anonymize all data (remove PHI)
4. ✅ Secure storage (encrypted)
5. ✅ Document data provenance
6. ✅ Plan for data destruction

**PHI to Remove:**
- Patient names, addresses, phone numbers
- Medical record numbers
- Dates (use relative times: "Day 0", "Day 7")
- Device serial numbers (scanners)
- Image metadata (EXIF, DICOM tags)

## 🏆 Recognition

Contributors will be acknowledged in:
- GitHub contributors list
- CHANGELOG.md
- Future publications (if substantial contribution)

## 📞 Questions?

- **Technical Issues:** Open a GitHub Issue
- **Security Concerns:** Email maintainer directly (not public)
- **Clinical Collaboration:** Contact PI for partnership opportunities

## 📜 Legal

By contributing, you agree that:
- Your contributions are your original work
- You have rights to license the contribution
- Your work is licensed under project's LICENSE
- You understand this is research software (not for clinical use)
- You will not make medical claims based on this software

---

**Thank you for contributing responsibly to medical AI research! 🙏**

Together we can advance healthcare technology while prioritizing patient safety.

---

**Last Updated:** November 24, 2025  
**Maintainer:** Emir
