# Requires 
# 1. kbatch python library
# 2. kbatch authentication to PC
# 3. AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_SAS_TOKEN env vars set

kbatch job submit -e "{\"AZURE_STORAGE_ACCOUNT\":\"$AZURE_STORAGE_ACCOUNT\",\"AZURE_STORAGE_SAS_TOKEN\":\"$AZURE_STORAGE_SAS_TOKEN\"}" --name $1 -f $2
