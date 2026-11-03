# Credit Risk Cost-Sensitive Decision Engine
**آزمایشگاه فناوری های مالی داتین- ترم سوم**  
**دانشجو:** شایان گنجی (Shayan Ganji)  

- **مستندات و گزارش‌های تصمیمات مهندسی:** [گزارش فارسی (report.fa.md)](report.fa.md) | [English Report (report.md)](report.md)
- **نوت‌بوک تحلیلی و روایی:** [نوت‌بوک اجرا شده (solution.ipynb)](solution.ipynb)

## Quick Start Guide
```bash
# 1. Install pinned dependencies in editable mode
pip install -e .

# 2. Run offline unit test suite (10 tests)
python -m unittest discover -s tests -p "test_*.py"

# 3. Launch narrative Jupyter Notebook (reads from PostgreSQL via SQLAlchemy)
jupyter notebook solution.ipynb
```
