from dask_gateway import Gateway

g = Gateway(
    address="https://pccompute.westeurope.cloudapp.azure.com/compute/services/dask-gateway",
    auth="jupyterhub",
)
g.list_clusters()
breakpoint()
[g.connect(c.name).shutdown() for c in g.list_clusters()]
