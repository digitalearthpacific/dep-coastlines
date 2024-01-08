import geopandas as gpd


countries = [
    "American Samoa",
    "Cook Islands",
    "Fiji",
    "French Polynesia",
    "Guam",
    "Kiribati",
    "Marshall Islands",
    "Micronesia",
    "Nauru",
    "New Caledonia",
    "Niue",
    "Northern Mariana Islands",
    "Palau",
    "Papua New Guinea",
    "Pitcairn Islands",
    "Solomon Islands",
    "Samoa",
    "Tokelau",
    "Tonga",
    "Tuvalu",
    "Vanuatu",
    "Wallis and Futuna",
]

all_polys = gpd.read_file(
    "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-gpkg.zip"
).query("NAME_0 in @countries")

all_polys.dissolve("NAME_0").to_file("data/aoi.gpkg")
