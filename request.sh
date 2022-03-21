#!/bin/sh
curl -X 'POST' \
'http://localhost:5000/predict-shows' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
    "age_range": {
        "range_start": 21,
        "range_end": 30
    },
    "user_interests": {
        "reading": "thriller",
        "travel": "backpacking",
        "sports": "table-tennis"
    },
    "user_languages": ["german", "french"]
    }'
