# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from fastapi import FastAPI
from pydantic import BaseModel
from recommendations import recommend_cars

app = FastAPI()

class CustomerInput(BaseModel):
    #brand: str
    model_group: str
    type: str
    fuel: str
    price: float
    #horsepower: float
    #OCCUPATIONGROUP: str
    #REGION: str

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
@app.post("/recommend")
def recommend(customer: CustomerInput):
    try:
        customer_dict = customer.dict()
        recommendations = recommend_cars(customer_dict)
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}


