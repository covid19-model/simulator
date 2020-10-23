for file in "$@"; do
    pdfcrop "$file" tmp.pdf && mv tmp.pdf "$file"
done
