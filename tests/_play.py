#%%
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tests.test_account_encoder import AccountEmbedderTester
from models.embeddings.account_encoder import AccountEncoder

# Initialize tester
tester = AccountEmbedderTester(
    database_name="liebre_dev",
    business_id="bu-651"
)

# Load data
tester.load_data()

# Train and test embedder
embedder = AccountEncoder(dim=10)
result = tester.test_embedder(
    embedder,
    distance_metric='euclidean'  # or 'hyperbolic_poincare', 'hyperbolic_lorentz'
)

# Print results
tester.print_test_summary(result)