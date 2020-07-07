to know how the code works refer "KnowledgeBase.docx" from InformationRepository directory
-----------------------------------------------------------------
To make the code working, user must have depenedent packages configured.

Dependencies are mentioned in the requirements.txt file

to Install dependencies, just run the follwoing command 

``` pip install -r requirments.txt ```

more details :
https://stackoverflow.com/questions/7225900/how-to-install-packages-using-pip-according-to-the-requirements-txt-file-from-a




Note: This programe uses Pytesseract to extract the text from the image. so pytesseract executble path should be configured in the config file.

pytesseract can be downloaded from 
https://github.com/tesseract-ocr/tesseract/wiki

Make sure English language packages are present in the file:
https://github.com/tesseract-ocr/tessdata/tree/3.04.00

downlaod all the language files and replace it in installed tesseract folder
