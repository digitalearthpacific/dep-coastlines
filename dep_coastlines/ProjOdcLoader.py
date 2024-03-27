from dep_tools.loaders import OdcLoader


class ProjOdcLoader(OdcLoader):
    # See https://github.com/digitalearthpacific/dep-coastlines/issues/34
    def load(self, items, areas):
        self._kwargs["crs"] = int(areas.iloc[0].epsg)
        return super().load(items, areas)
