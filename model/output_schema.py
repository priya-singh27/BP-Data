from pydantic import BaseModel

class BPOutputSchema(BaseModel):
    SYSTOLIC: int
    SYSTOLIC_UNIT: str
    DIASTOLIC: int
    DIASTOLIC_UNIT: str
    PULSE: int
    PULSE_UNIT: str
