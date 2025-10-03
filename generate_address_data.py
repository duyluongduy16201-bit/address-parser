import json

json_path = "genai_public_standard_address_vn.json"
output_path = "address_data_json_based.py"

with open(json_path, encoding="utf-8") as f:
    mapping = json.load(f)

province_name_to_id = {}
district_key_to_id = {}
ward_key_to_id = {}

PROVINCES, DISTRICTS, WARDS = [], [], []
PROVINCE_TO_DISTRICTS = {}
DISTRICT_TO_WARDS = {}

prov_counter, dist_counter, ward_counter = 1, 1, 1

for rec in mapping:
    pname = rec.get("province")
    dname = rec.get("district")
    wname = rec.get("ward")

    # Province
    if pname and pname not in province_name_to_id:
        province_name_to_id[pname] = prov_counter
        PROVINCES.append((prov_counter, pname))
        PROVINCE_TO_DISTRICTS[prov_counter] = []
        prov_counter += 1
    pid = province_name_to_id.get(pname)

    # District (key theo province)
    if dname:
        district_key = (pid, dname)
        if district_key not in district_key_to_id:
            district_key_to_id[district_key] = dist_counter
            DISTRICTS.append((dist_counter, dname, pid))
            PROVINCE_TO_DISTRICTS[pid].append(dist_counter)
            DISTRICT_TO_WARDS[dist_counter] = []
            dist_counter += 1
        did = district_key_to_id[district_key]
    else:
        did = None

    # Ward (key theo district)
    if wname and did:
        ward_key = (did, wname)
        if ward_key not in ward_key_to_id:
            ward_key_to_id[ward_key] = ward_counter
            WARDS.append((ward_counter, wname, did))
            DISTRICT_TO_WARDS[did].append(ward_counter)
            ward_counter += 1

# Xuất ra file
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# -*- coding: utf-8 -*-\n")
    f.write("# Generated address data (with mappings)\n\n")

    f.write("PROVINCES = [\n")
    for item in PROVINCES:
        f.write(f"    {item},\n")
    f.write("]\n\n")

    f.write("DISTRICTS = [\n")
    for item in DISTRICTS:
        f.write(f"    {item},\n")
    f.write("]\n\n")

    f.write("WARDS = [\n")
    for item in WARDS:
        f.write(f"    {item},\n")
    f.write("]\n\n")

    f.write("PROVINCE_TO_DISTRICTS = {\n")
    for pid, dlist in PROVINCE_TO_DISTRICTS.items():
        f.write(f"    {pid}: {dlist},\n")
    f.write("}\n\n")

    f.write("DISTRICT_TO_WARDS = {\n")
    for did, wlist in DISTRICT_TO_WARDS.items():
        f.write(f"    {did}: {wlist},\n")
    f.write("}\n")

print(f"Generated {len(PROVINCES)} provinces, {len(DISTRICTS)} districts, {len(WARDS)} wards.")
print(f"Saved to {output_path}")
