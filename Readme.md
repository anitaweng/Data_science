1. Install requirements
    ```
    pip install -r requirements.txt
    ```
2. For problem 1, implementing ppt beauty crawler for whole 2020.
   --find the top 10 pusher article (2nd)
   --find the article which are pushed over 100 times (3rd)
   --find the images in the articles which contain the specific keywords (4th)
   ```
   # for crawl the 2020 articles
   python 309505001.py crawl
   ```
   ```
   # for the 2nd function
   python 309505001.py push start_date end_dat
   ```
   ```
   # for the 3rd function
   python 309505001.py popular start_date end_date
   ```
   ```
   # for the 4th function
   python 309505001.py keyword {keyword} start_date end_date
   ```
3. For problem 2, given an image then predict whether it is popular article or not.
    ```
    python 309505001.py imgfilelistname
    ```
