WORK IN PROGRESS
 
 
Introduction
=============
This reporitory includes the code for our trend detection and forecasting model, attend2trend. Furthermore, the repository provides a framework for running, and comparing, different statistical-based trend detection models.  

Prerequisites
=============
The required software packages will be installed by the setup.py file 

 python setup.py install


General Framework and Simple Guide: 
===================================
In this section, we provide a brief describtion of the three main building blocks of our framework: 

**(1) Generating a time series For the detected entities:** 
We extract the entities from the textual datasets (e.g. Twitter stream), with their actual time stamp. Then, we store them in a seperate file. 

Input:  textual dataset (e.g. twitter stream) 

Output: CSV file that contains the following columns (Entity, Time1, Time2, ...)

Directory: trendify/trendify/create_entity_timestamps_csv.py 

Commands:

 python create_entity_timestamps_csv.py + parameters 

 -i "input_path_to_twitter_tsvs" (default "/data21/asaleh/share/Tweet2011/tweets")
 -m "NER model to be used" (default="config_twitter_wnut16")
 -o "output_csv_file" (default="./entity_timestamps.csv")


**(2) Convert the entites time series to Frequency time series:**
In this step, we will use the CSV which have been generated from the last step.

Input: Time step (hourly or daily) and the directory of the CSV file

Output: For each entity, create a frequency time series, like ( Trump, 15, 0, 0, 13, 14, 12, 100, ......) . The output is also stored in a seperate file. We will need to apply some threshold to remove the entities which doesn't have enough data.

Directory: trendify/trendify/create_frequency_time_series.py

Commands: python create_frequency_time_series.py + parameters

 -i "input entity timestamps csv file created in the previous script" (default="./entity_timestamps.csv")
 -t "timestep to be used" (default="hour")
 -th "threshold e.g. the minimum amount an entity has to occur in the whole date range" (default=0)
 -o "output_csv_file" (default="./frequency_time_series_data.csv")


**(3) Detect the trends in the frequency time series:**
The main code of the model is placed in the following file: 
attention_model_forecast_v1.py

The entrypoint function is named "trendify2" in the setup.py file. The most important parameters when calling trendify2 are: 

-r       - which is the path to the file, where the results and scores are then saved
-mo      - is used to specify the model output directory, where the trained model will be saved
-data    - this is the directory with the training data. it expects 3 files there: train, test, valid
-a       - activates or deactivates attention via values "y" (attention), "n" (no attention), "both" (models with attention and without attention will be trained). Defaults to both
-atlas   - Here you can set how many values before an atlas trend should be marked and used.

The existing model can be loaded using the -model parameter. In order to load the model, you have to have the right parameters set in the config generator.


Useful code comments: 
=====================
- **General framework parameters:** If you would like to change hyperparameters and parameters, you have to modify code in the "generate_parameter_configs" method.
- **Grid search:** There are multiple arrays ending with _possible which have all the possible parameter configurations, so that you can perform a grid search to find the best ones if you want.
- **Comparing the results with EbayAtlas Trend:** After training the networks, the framework will also evaluate run the statistical models and use it for the results comparison (you can comment out this feature from line 474 "evaluate_with_statistical_models(test_generator, args.results_path)").
- **Early stopping during the model training:** The model will be saved after each epoch and early stopping is activated after 10 epochs of no improvements of the validation loss. 
- **Default GPU usage:** In line 61, you find 'gpu_memory_fraction = 0.3' where you can set the fraction of gpu memory allocation. TensorBoard can also be used using the created logs directory.








