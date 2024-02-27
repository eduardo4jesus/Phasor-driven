for pdf in *.pdf; do
  pdftoppm "$pdf" "${pdf%.pdf}" -png
done
