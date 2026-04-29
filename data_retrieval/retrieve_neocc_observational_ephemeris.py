# %%
from tudatpy.data.esaneocc import NEOCCQuery
import matplotlib.pyplot as plt
from tudatpy.estimation.observations.observations_adapter.observation_collection_adapter import UniversalObservationAdapter

# %%
# Define parameters
designator = '433'
start_epoch = "2000-01-01 00:00"
end_epoch = "2000-05-01 10:00"
stepsize = 1
stepsize_unit = "weeks"
center = '500'

# %%
# Fetch the raw data using esaneocc service
raw_df = NEOCCQuery.get_observational_ephemerides(
    designator, start_epoch, end_epoch, stepsize, stepsize_unit, center
)

# %%
# Instantiate the Adapter
universal_observation_collection_adapter = UniversalObservationAdapter()
# Use the adapter to create a Wrapper Instance
# This returns an OBJECT of type TudatObservationWrapper
observation_wrapper = universal_observation_collection_adapter.from_df(raw_df, designator, center)

# %%
# Call visualize
observation_wrapper.visualize(projection='mollweide')
plt.show()
