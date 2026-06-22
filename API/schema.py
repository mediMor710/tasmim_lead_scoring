from pydantic import BaseModel, Field
from typing import Literal

class LeadRequest(BaseModel):
    """
    Defines what data should the API expect when scoring a lead.
    
    Pydantic BaseModel does 3 things automtically:
    1. Checks the incoming data has the right type.
    2. Determines the allowed values a field should accept.
    3. Generates API Documentation.
    """

    full_name: str = Field(..., example='John Smith')
    email: str = Field(..., example='john@company.com')
    company_name: str = Field(..., example='TechCorp')
    # Literal forces the user to pick one of the specific values
    company_size: Literal['solo', 'small', 'medium', 'large'] = Field(
        ..., example='medium'
    )
    service_type: Literal['website', 'ecommerce', 'mobile_app', 'branding', 'seo'] = Field(
        ..., example='ecommerce'
    )
    budget_range: Literal['low', 'medium', 'high', 'enterprise'] = Field(
        ..., example='high'
    )
    deadline: Literal['urgent', 'normal', 'flexible'] = Field(
        ..., example='urgent'
    )
    contact_channel: Literal['email', 'whatsapp', 'phone', 'social_media'] = Field(
        ..., example='whatsapp'
    )
    message_text: str = Field(
        ..., example='We need a full e-commerce platform with payment integration.'
    )

class LeadResponse(BaseModel):
    """
    Defines what the API returns after scoring process.
    """
    
    full_name: str
    score: float
    priority: str
    conversion_probability: float
    top_factors: dict   # Top 5 SHAP explanations
    recommendation: str