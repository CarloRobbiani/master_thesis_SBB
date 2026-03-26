import torch
from graph_models.station_graph.stationMATGCN import StationMATGCN
from simulator.simulation import Simulator
from utils import PolicyWrapper, sample_actions, compute_laplacian

# 1. Load your trained MATGCN model
matgcn = StationMATGCN(
    num_features=...,           # Number of train features
    external_features=...,      # Number of external (weather) features
    hidden_dim= 64,             # Hidden dimension size
    K= 3,                      # Graph convolution order
    num_blocks= 2,             # Number of STBlocks
    horizon=...                 # Prediction horizon
)
matgcn.load_state_dict(torch.load('path_to_trained_model.pt'))
matgcn.eval()

# 2. Prepare the simulator
sim = Simulator(
    model=matgcn,
    deltat=300,                 # e.g., 5 minutes
    column_mapping=...,         # Dict mapping feature names to indices
    cat_cols_md=...,            # Categorical columns metadata
    stations_emb=...,           # Station embeddings
    lines_emb=...,              # Line embeddings
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    nb_past_stations=...,
    nb_future_stations=...,
    embedding_size=...,
    idle_time_end=...,
    net_type="mlp",             # DO NOT USE MATGCN WILL BREAK AS NOT IMPLEMENTED IN SIMULATION
    #net_type='matgcn',         # Or whatever you use to identify MATGCN
    local_features=True         # If you want local features
)

# 3. Prepare initial states and metadata
states = [...]                 # List of initial state tensors
states_metadata = [...]        # List of metadata for each train
states_time = [...]            # List of initial timestamps
itineraries = {...}            # Dict of itineraries
nb_steps = ...                 # Number of simulation steps
nb_samples = ...               # Number of stochastic samples

# 4. Run the simulation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PolicyWrapper(matgcn, T=12, device=device)

sim_states, sim_metadata, sim_times, sim_manager = sim.init_simulation(
    states, states_metadata, states_time,
    nb_steps, nb_samples, itineraries, mode='predict'
)

policy.reset()

for step in range(nb_steps):

    # ✅ get state from simulator
    current_states = sim_manager.states            # [B, N, F]
    padding_mask = sim_manager.padding_mask        # [B, N]

    # ⚠️ you must define this once
    adj = [0][0]
    laplacian = compute_laplacian(adj).to(device)

    with torch.no_grad():
        logits = policy(current_states, laplacian)

    actions = sample_actions(logits, padding_mask)

    # ✅ correct simulator usage
    sim_manager.update_positions(actions)
    sim_manager.update(actions)

# 5. Collect and analyze results
#results = sim_manager.output  # Or however your simulator stores results
delay_idx = sim_manager.group_features_mapper['PAST_DELAYS']
delays = sim_manager.states[:, :, delay_idx[-1]]