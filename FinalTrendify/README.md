
## File Structure

    Trendify
        - Data
            - twitter-data
            - wiki-data
        - others
            - wiki_data_creation.py
        - Results
            - twitter-results
            - wiki-results
        - trendify
            - Main
                - main.py
                - old_code.py
            - LSTM
                - lstm_general.py
                - old_code
            - Statistical-Methods
                - ARIMA
                    - arima_general.py
                    - old_code
                - EBAY
                    - ebay_general.py
                    - old_code
                - MA
                    - ma_general.py
                    - old_code
                - SARIMA
                    - sarima_general.py
                    - old_code
                - TES
                    - tes_general.py
                    - old_code
        - README.md
        - requirements.txt
        - setup.py
---



### Entry point

    - trendify/Main/main.py


### create virtual environment or enable it
    - python3 -m venv /path/to/new/virtual/environment
    - eg., python3 -m venv pyenv3.8

### activate virtual environment
    - source /path/to/new/virtual/environment/bin/activate
    - eg., source pyenv3.8/bin/activate

### install requirements.txt (activate venv before)
    - pip install -r requirements.txt

### export PYTHONPATH before running main.py script
    - export PYTHONPATH="${PYTHONPATH}:path/to/root/directory/of/project"
    - eg., export PYTHONPATH="${PYTHONPATH}:." (where . represents current directory)


### Run main file
    - goto the root directory of the project 
    - PWD: /Trendify/
    - activate virtual environment
    - python3 trendify/Main/main.py 
        -- optional arguments based on reuirements
        -h, --help            show this help message and exit
        --file_name DATA_FILE_NAME
                        frequency time series csv data file name
        --data_dir DATA_DIR_NAME
                        data directory name in ./Data
        --methods METHODS [METHODS ...]
                        list of methods want to apply on the data from
                        ['EBAY', 'MA', 'ARIMA', 'SARIMA', 'TES', 'LSTM']
        --result_dir RESULT_DIR_NAME
                        result directory name in ./Results
        --computation DO_COMPUTATION
                        True: computation of trend_points + analysis, False: only analysis


### BEFORE Running/Traning 
    - put a csv timeseries data file of below format in Data/your-data/ directory



### eg., dataname = new_data


| "words" | t1 | t2 | t3 |
| ------- | --- | --- | --- |
| "word1" | 20 | 35 | 25 |
| "word2" | 24 | 31 | 23 |
| "word3" | 23 | 30 | 25 |
| "word4" | 24 | 39 | 29 |

### --file_name
    - the filename of csv data file
    - eg., "sample_10.csv"

### --data_dir
    - the --data_dir will get directory-name of the data you put in
    - eg., twitter-data with sample_10.csv in it.

### --methods
    - list of methods want to apply on the data from ['EBAY', 'MA', 'ARIMA', 'SARIMA', 'TES', 'LSTM']
    - eg., can be combination of this ['MA', 'ARIMA']

### --result_dir
    - results directory name where results will be stored
    - eg., --result_dir=new_data_results

### --computation
    - True: computation of trend_points + analysis, False: only analysis



## some of the examples (after activating venv)
    - python3 trendify/Main/main.py (with default arguments)
    - python3 trendify/Main/main.py --file_name=new_amazon_data.csv --methods=EBAY LSTM ARIMA MA --result_dir=AMAZON_RESULT 




    
        
    


