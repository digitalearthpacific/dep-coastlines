# Requires 
# 1. kbatch python library
# 2. kbatch authentication to PC
# 3. AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_SAS_TOKEN env vars set
kbatch job submit -f src/kbatch_calculate_rasters.yml -e "{\"AZURE_STORAGE_ACCOUNT\":\"$AZURE_STORAGE_ACCOUNT\",\"AZURE_STORAGE_SAS_TOKEN\":\"$AZURE_STORAGE_SAS_TOKEN\"}" --name x
