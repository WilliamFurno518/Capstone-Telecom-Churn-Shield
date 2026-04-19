"""
Canonical feature schema shared by the mapping and modelling layers.

These names are the *internal* language of ChurnShield. The MappingService
translates whatever columns a user uploads into this schema; everything
downstream (insights, misalignment engine) assumes these names.
"""

CORE_FEATURES = {
    "customer_id": {
        "description": "Unique customer identifier",
        "type": "identifier",
        "required": True,
    },
    "phone": {
        "description": "Customer phone number",
        "type": "identifier",
        "required": True,
    },
    "current_plan": {
        "description": "Current subscription plan or product",
        "type": "categorical",
        "required": True,
    },
    "monthly_cost": {
        "description": "Monthly bill or fee amount",
        "type": "numerical",
        "required": True,
    },
    "usage_primary": {
        "description": "Primary usage metric (data GB, balance, claims, ...)",
        "type": "numerical",
        "required": True,
    },
    "usage_secondary": {
        "description": "Secondary usage metric (calls, transactions, ...)",
        "type": "numerical",
        "required": True,
    },
    "usage_tertiary": {
        "description": "Tertiary usage metric (SMS, digital logins, ...)",
        "type": "numerical",
        "required": True,
    },
    "complaints": {
        "description": "Number of complaints or support tickets",
        "type": "numerical",
        "required": True,
    },
    "tenure": {
        "description": "Account age, typically in months",
        "type": "numerical",
        "required": True,
    },
    "extra_usage": {
        "description": "Additional usage metric (intl calls, transfers, ...)",
        "type": "numerical",
        "required": False,
    },
}
