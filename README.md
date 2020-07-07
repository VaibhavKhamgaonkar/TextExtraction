# TextExtraction
Extracting the Text from Restaurants bills
------------------------------------------------------------------------
Refer "KnowledgeBase.docx" from InformationRepository directory for more details.

To make the code working, user must have depenedent packages configured.
Dependencies are mentioned in the requirements.txt file

to Install dependencies, just run the follwoing command 

``` pip install -r requirments.txt ```

more details :
https://stackoverflow.com/questions/7225900/how-to-install-packages-using-pip-according-to-the-requirements-txt-file-from-a


Steps:
1. Place all the Data Files images in the Data Folder
2. Configure the parameter in the config file
3. Run the Python file  : Main.py using this command  from working directory
``` python Main.py ```






Note: This programe uses Pytesseract to extract the text from the image. so pytesseract executble path should be configured in the config file.

pytesseract can be downloaded from 
https://github.com/tesseract-ocr/tesseract/wiki

Make sure English language packages are present in the file:
https://github.com/tesseract-ocr/tessdata/tree/3.04.00

downlaod all the language files and replace it in installed tesseract folder
