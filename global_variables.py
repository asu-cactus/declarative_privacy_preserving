date_range = 365
location_range = 2
countries = ['UK', 'US', 'BRZ', 'CHN', 'AU']

hidden_units = 500

frequency_range = 1
training_size = 1000

batch_size = 100
learning_rate = 0.002
epochs = 3000

l2_norm_clip = 1
# noise_multiplier = 0.03
noise_multiplier = 0.3

sigma = 0.1
clip = 0.5
delta = 1e-5

assert training_size % frequency_range == 0
assert frequency_range < location_range
assert frequency_range < date_range


airport_locations = [
    "Buffet, Phoenix Sky Harbor Airport",                # 0
    "Smoking Area, Phoenix Sky Harbor Airport",          # 1
    "Baggage Reclaim Area, Phoenix Sky Harbor Airport",  # 2
    "Waiting Area, Phoenix Sky Harbor Airport",          # 3
    "Reading Area, Phoenix Sky Harbor Airport",          # 4
    "Recreation Area, Phoenix Sky Harbor Airport",       # 5
    "Food Court, Phoenix Sky Harbor Airport",            # 6
    "Lobby, Phoenix Sky Harbor Airport"                  # 7
]
