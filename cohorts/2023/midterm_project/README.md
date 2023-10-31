### Hotel bookings
We have data from kaggle That contains basic information about hotel bookings. For example, it contains information about how many adults and children were in the booking, what was the source of the booking, what type of room was booked, and so on. The key columns that we are trying to determine or whether the booking was canceled or not. The goal is to create a model that can predict whether a booking has a high chance of being canceledThat contains basic information about hotel bookings. For example, it contains information about how many adults and children were in the booking, what was the source of the booking, what type of room was booked, and so on. The key columns that we are trying to determine or whether the booking was canceled or not. The goal is to create a model that can predict whether a booking has a high chance of being canceled..

### Data
The source of the data is https://www.kaggle.com/datasets/rajatsurana979/hotel-reservation-data-repository?select=hotel_booking.csv
and it is stored in hotel_booking.csv

Notebook.ipynb Is used to do the exploratory data analysis correlation and create the code for the model and the hyper parameter tuning.

predict.py - This is the file that will run on the FastAPI server and serve the model.

The environment, isolation was done using pipenv

Deployment is done using GCS and Google cloud run