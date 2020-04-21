pdoc --pdf nf > docs.md

pandoc --metadata=title:"Neural Framework Documentation" \
    --toc --toc-depth=4 --from=markdown+abbreviations \
    --pdf-engine=xelatex \
    --output=docs.pdf docs.md

rm docs.md