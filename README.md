# چالش سوم: طبقه‌بندی حساس به هزینه و موتور تصمیم‌گیری ریسک اعتباری
**دانشجو:** شایان گنجی  
**آزمایشگاه:** آزمایشگاه فناوری های مالی داتین- ترم سوم  
**پایگاه‌داده هدف:** PostgreSQL 16.4 | دیتابیس `olist_db` (اسکیمای `credit`)  

---

**مستندات و گزارش‌های مهندسی:**
- [نسخه فارسی گزارش تحلیلی و تصمیم‌های مهندسی (report.fa.md)](report.fa.md)
- [English Technical Architecture & Decision Report (report.md)](report.md)
- [نوت‌بوک تحلیلی اجرا شده (solution.ipynb)](solution.ipynb)

---

## راهنمای اجرا و فرامین خط فرمان (CLI)

```bash
# ۱. نصب بسته در حالت توسعه
pip install -e .

# ۲. اجرای آزمون‌های واحد آفلاین (۱۰ تست مستقل)
python -m unittest discover -s tests -p "test_*.py" -v

# ۳. آموزش مدل و ارزیابی ماتریس هزینه
python -m src.credit_risk.cli --train-and-evaluate

# ۴. اجرای پیش‌بینی روی متقاضیان نمونه
python -m src.credit_risk.cli --predict

# ۵. اجرای نوت‌بوک تحلیلی در محیط ژوپیتر
jupyter notebook solution.ipynb
```

---

## ساختار مخزن پروژه

```text
stage-3-shayan-ganji/
├── solution.ipynb                  # نوت‌بوک تحلیلی اجرا شده (فراخوانی از دیتابیس)
├── report.fa.md                    # گزارش جامع مهندسی، تصمیم‌ها و مصالحه‌ها (فارسی)
├── report.md                       # گزارش جامع معماری و کارایی (انگلیسی)
├── pyproject.toml                  # تعاریف وابستگی‌ها و بسته‌بندی پروژه
├── README.md                       # راهنمای اجرایی مختصر
├── models/
│   └── credit_decision_engine.joblib # آرتیفکت ذخیره‌شده پایپ‌لاین و متادیتا
├── src/
│   └── credit_risk/
│       ├── config.py               # تنظیمات پایگاه‌داده، پارامترها و نسبت‌های هزینه
│       ├── data_loader.py          # واکشی داده از ویوی تحلیلی دیتابیس با SQLAlchemy
│       ├── feature_engineering.py  # ترنسفورمرهای ویژگی‌های قلمرو و آزمون عدم نشت
│       ├── pipeline.py             # پایپ‌لاین‌های یادگیری ماشین (سایکیت‌لرن)
│       ├── cost_evaluation.py      # ارزیابی ماتریس هزینه، بهینه‌سازی آستانه و CV
│       ├── model_card.py           # مستندسازی و شناسنامه فنی مدل
│       └── cli.py                  # واسط خط فرمان آموزش، ارزیابی و استنتاج
└── tests/
    └── test_credit_risk.py         # آزمون‌های واحد آفلاین (تست‌های ۱۰گانه)
```
