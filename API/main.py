from fastapi import FastAPI, HTTPException
from datetime import datetime
import sys, os
import uvicorn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from API.schema import LeadRequest, LeadResponse
from API.predictor import score_lead

##### Create the FastAPI app #####
app = FastAPI(
    title='Tasmim Web - Lead Scoring API',
    description='Scores incoming leads and predicts conversion probability.',
    version='1.0.0'
)

##### Health check endpoint #####
@app.get('/health') # creates a URL endpoint
def health_check():
    return {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'model': 'Random Forest',
        'version': '1.0.0'
    }

##### Score one lead #####
@app.post('/score', response_model=LeadResponse)
async def score_single_lead(lead: LeadRequest):
    """
    Scores a single lead and returns conversion probability.
    Send a POST request with lead details, get a score back.
    """

    try:
        # Converts a Pydantic object into a Python dictionary
        lead_dict = lead.model_dump()
        result = score_lead(lead_dict)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
##### Score multiple leads at once #####
@app.post('/score/batch')
async def score_batch(leads: list[LeadRequest]):
    """
    Scores multiple leads in one request.
    Returns a list of results sorted by score.
    """

    if len(leads) > 100:
        raise HTTPException(
            status_code=400,
            detail='Maximum 100 leads per batch request.'
        )
    try:
        results = []
        for lead in leads:
            lead_dict = lead.model_dump()
            result = score_lead(lead_dict)
            results.append(result)

        results.sort(
            key=lambda result: result['score'],
            reverse=True
        )

        return {
            'total': len(results),
            'leads': results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
##### Running the server #####
if __name__ == '__main__':
    
    uvicorn.run(
        'API.main:app',
        host='0.0.0.0',
        port=8000,
        reload=True
    )
