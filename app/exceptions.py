from fastapi import HTTPException, status

class ModelNotLoadedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Model not loaded or unavailable'
        )

class InvalidInputException(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid input: {message}'
        )

class PredictionException(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Prediction failed: {message}'
        )
