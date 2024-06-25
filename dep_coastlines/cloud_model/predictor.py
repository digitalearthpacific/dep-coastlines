from dea_tools.classification import predict_xr
from xarray import concat

from dep_coastlines.cloud_model.fit_model import SavedModel


class ModelPredictor:
    def __init__(self, model: SavedModel):
        self.model = model
        self.codes = self.model.codes.groupby(self.model.response_column).first()

    def code_for_name(self, name):
        return (
            self.model.codes.reset_index()
            .set_index("code")
            .loc[name, self.model.response_column]
        )

    def calculate_mask(self, input):
        masks = []
        for year in input.year:
            prediction = predict_xr(
                self.model.model,
                input.sel(year=year)[self.model.predictor_columns],
                clean=True,
                proba=True,
            )
            prediction.coords["year"] = year
            masks.append(prediction)
        return concat(masks, dim="year")

    def apply_mask(self, input):
        cloud_code = self.code_for_name("cloud")
        if isinstance(input, list):
            mask = self.calculate_mask(input[0])
            output = input[0].where(mask.Predictions != cloud_code, drop=False)
            for ds in input[1:]:
                ds = ds.sel(year=ds.year[ds.year.isin(output.year)])
                this_mask = self.calculate_mask(ds)
                missings = (output.isnull()) | (output["count"] <= 4)
                mask = mask.where(
                    mask.Predictions != cloud_code,
                    this_mask.where(this_mask.Predictions != cloud_code, drop=False),
                )
                output = output.where(
                    ~missings, ds.where(this_mask.Predictions != cloud_code, drop=False)
                )
        else:
            mask = self.calculate_mask(input)
            output = input.where(mask != cloud_code)
        return output, mask
