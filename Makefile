build:
	docker build . --tag digitalearthpacific/coastlines:local

data/aoi.gpkg:
	python3 src/aoi/aoi.py

data/coastal_pathrows.geojson:
	python3 src/aoi/pathrows.py
