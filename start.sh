#!/bin/bash
# start.sh - تشغيل تطبيق Flask على Render

# تفعيل البيئة الافتراضية إذا كانت موجودة (اختياري)
# source venv/bin/activate

# تعيين متغيرات البيئة الضرورية
export FLASK_APP=app_advanced_with_tda_fixed.py
export FLASK_ENV=production
export PYTHONUNBUFFERED=1

# تأكد من أن جميع الحزم مثبتة
pip install --upgrade pip
pip install -r requirements.txt

# تشغيل التطبيق على المنفذ الذي يحدده Render
# Render يحدد PORT تلقائياً عبر متغير البيئة $PORT
if [ -z "$PORT" ]; then
    PORT=5000
fi

# تشغيل Flask
gunicorn --bind 0.0.0.0:$PORT app_advanced_with_tda_fixed:app
