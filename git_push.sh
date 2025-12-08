#!/bin/bash
# Hızlı git push scripti

echo "Git status:"
git status

echo ""
read -p "Commit mesajı: " commit_message

if [ -z "$commit_message" ]; then
    echo "Hata: Commit mesajı boş olamaz!"
    exit 1
fi

git add .
git commit -m "$commit_message"
git push

echo ""
echo "✓ Değişiklikler GitHub'a push edildi!"
