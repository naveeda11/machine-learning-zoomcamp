import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

import json
app = FastAPI()
with open("dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)

# Load model from file
model = XGBClassifier()
model.load_model('xgb_model.json')

class Customer(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    meal: str
    country_code: str 
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: float
    company: str
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    reservation_status: str
    reservation_status_date: str
    name: str
    email: str
    phone_number: str
    credit_card: str

class Response(BaseModel):
     y_pred: float
     cancellation: bool

@app.post("/predict")
async def predict(customer: Customer) -> Response:   
    customer_dict = json.loads(customer.json())
    X = dv.transform([customer_dict])
    y_pred = model.predict_proba(X)[0, 1]
    cancellation = y_pred >= .5
    return Response(y_pred = y_pred, cancellation = bool(cancellation))

 
# customer = {'hotel': 'City Hotel',
#  'lead_time': 28,
#  'arrival_date_year': 2016,
#  'arrival_date_month': 'October',
#  'arrival_date_week_number': 44,
#  'arrival_date_day_of_month': 25,
#  'stays_in_weekend_nights': 0,
#  'stays_in_week_nights': 3,
#  'adults': 2,
#  'children': 0.0,
#  'babies': 0,
#  'meal': 'BB',
#  'country code': 'PRT',
#  'market_segment': 'Online TA',
#  'distribution_channel': 'TA/TO',
#  'is_repeated_guest': 0,
#  'previous_cancellations': 0,
#  'previous_bookings_not_canceled': 0,
#  'reserved_room_type': 'A',
#  'assigned_room_type': 'A',
#  'booking_changes': 0,
#  'deposit_type': 'No Deposit',
#  'agent': 9.0,
#  'company': 0,
#  'days_in_waiting_list': 0,
#  'customer_type': 'Transient',
#  'adr': 129.0,
#  'required_car_parking_spaces': 0,
#  'total_of_special_requests': 0,
#  'reservation_status': 'Check-Out',
#  'reservation_status_date': '28-10-2016',
#  'name': 'Patricia Martin',
#  'email': 'Patricia_M@att.com',
#  'phone-number': '581-432-0373',
#  'credit_card': '************5314'}