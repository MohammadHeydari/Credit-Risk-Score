from pydantic import BaseModel
from typing import Literal

class CreditRequest(BaseModel):
    status_checking_account: str
    duration_months: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_account: str
    employment_since: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    residence_since: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    number_existing_credits: int
    job: str
    number_dependents: int
    telephone: str
    foreign_worker: str