from coastlines.validation import deacl_val_stats
import geopandas as gpd
import numpy as np
import pandas as pd
import plotnine as pn
from sklearn.metrics import r2_score

from annual_validation import load_validation_data


def calculate_roc_stats(sets):
    def _stats_for_set(set_name, d):
        stats = dict(
            n_total=len(d),
            mae=d.mae.mean(),
            rmse=d.mse.mean() ** 0.5,
            sd=d.mae.std(),
            r2=r2_score(d.val_rate_time, d.cl_rate_time),
            n_sites=len(d.set_id.unique()),
        )

        plot = (
            pn.ggplot(d, pn.aes(x=d.val_rate_time, y=d.cl_rate_time))
            + pn.geom_point(size=0.5)
            + pn.theme_bw()
            + pn.labs(x="Validation ROC (m/yr)", y="DEP Coastlines ROC (m/yr)")
            #            + pn.annotate(
            #                "text",
            #                ha="left",
            #                label=f"n={stats['n_total']:,} @ {stats['n_sites']} sites\nr-sq={stats['r2']:.2f}\nMAE={stats['mae']:.2f} m/yr\nRMSE={stats['rmse']:.2f} m/yr",
            #                x=-25,
            #                y=25,
            #            )
        )
        plot.save(f"docs/{set_name}_rates_of_change_validation.png", dpi=300)

        plot_by_set = (
            pn.ggplot(d, pn.aes(x=d.val_rate_time, y=d.cl_rate_time, fill=d.set_id))
            + pn.geom_point()
            + pn.theme_bw()
            + pn.labs(x="Validation ROC", y="DEP Coastlines ROC")
        )
        plot_by_set.save(
            f"docs/{set_name}_rates_of_change_validation_by_set.png", dpi=300
        )

        return stats

    return pd.DataFrame.from_records(
        [_stats_for_set(name, df) for name, df in sets.items()], index=sets.keys()
    )


def annual_validation_summary():
    remove_bias = False
    validation_lines = gpd.read_file("data/validation/annual_lines_of_difference.gpkg")

    set_stats = validation_lines.groupby("set_id").apply(
        lambda set: deacl_val_stats(
            # Because we don't have a baseline dataset, the val_dist is always zero
            # (and therefore the correlation is undefined as well)
            np.zeros(len(set)),
            set["diff"],
            remove_bias=remove_bias,
        )
    )
    total_stats = deacl_val_stats(
        np.zeros(len(validation_lines)),
        validation_lines["diff"],
        remove_bias=remove_bias,
    )
    stats = pd.concat([set_stats, total_stats.to_frame(name="all").T])
    stats.to_csv("data/validation/annual_stats.csv")


def calculate_site_centroids():
    # For mapping
    def summarise_set(set):
        return gpd.GeoDataFrame(
            data=dict(
                geometry=[set.geometry.union_all().centroid],
            ),
            geometry="geometry",
            crs=set.crs,
        )

    load_validation_data().groupby("set_id").apply(
        summarise_set, include_groups=False
    ).reset_index().to_file("data/validation/site_centroids.gpkg")


def calculate_length_of_validation_lines():
    validation_data = load_validation_data()
    # JA did it all
    roc_data = validation_data[validation_data.digitiser_name.isna()]
    number_of_years = 8
    km_per_meter = 1 / 1000
    total_km = roc_data.geometry.length.sum() * km_per_meter
    print(f"Total length of roc validation data: {total_km:.2f}")
    print(f"KM per year: {total_km / number_of_years:.2f}")


def main():
    roc_data = gpd.read_file("data/validation/rates_of_change_validation.gpkg")
    roc_sets = dict(
        all_transects=roc_data, significant_trends=roc_data[roc_data.cl_sig_time < 0.01]
    )
    roc_stats = calculate_roc_stats(roc_sets)
    roc_stats.to_csv("data/validation/roc_validation_summary.csv", index=True)
    annual_validation_summary()
    calculate_site_centroids()
    calculate_length_of_validation_lines()


if __name__ == "__main__":
    main()
