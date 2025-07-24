{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a7ed5786",
   "metadata": {},
   "outputs": [],
   "source": [
    "import xml.etree.ElementTree as ET\n",
    "from xopen import xopen\n",
    "import gzip\n",
    "import polars as pl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "14c4bd0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "PLAN_PATH = \"/Users/andre/Desktop/Cergy/MATSim/matsim-berlin/berlin-v6.4.output_plans.xml.gz\"\n",
    "EVENTS_PATH = \"/Users/andre/Desktop/Cergy/MATSim/matsim-berlin/berlin-v6.4.output_events.xml.gz\"\n",
    "OUTPUT_DIR = \"/Users/andre/Desktop/Cergy/Python_Scripts/runs/fixed_10pct/output/\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a07f3656",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_passages():\n",
    "    records = []\n",
    "\n",
    "    with gzip.open(EVENTS_PATH, 'rt', encoding='utf-8') as f:\n",
    "        for event, elem in ET.iterparse(f, events=(\"end\",)):\n",
    "            if elem.tag == \"event\":\n",
    "                event_type = elem.attrib.get(\"type\")\n",
    "                if event_type in [\"entered link\", \"left link\"]:\n",
    "                    time = float(elem.attrib.get(\"time\", 0))\n",
    "                    records.append({\n",
    "                        \"time\": time,\n",
    "                        \"link_id\": elem.attrib.get(\"link\"),\n",
    "                        \"vehicle\": elem.attrib.get(\"vehicle\"), \n",
    "                        \"event_type\": event_type\n",
    "                        })\n",
    "\n",
    "                if event_type in [\"departure\", \"arrival\"]:\n",
    "                    time = float(elem.attrib.get(\"time\", 0))\n",
    "                    records.append({\n",
    "                        \"time\": time,\n",
    "                        \"link_id\": elem.attrib.get(\"link\"),\n",
    "                        \"vehicle\": elem.attrib.get(\"person\"), \n",
    "                        \"event_type\": event_type\n",
    "                    })\n",
    "            elem.clear()\n",
    "            \n",
    "    passages_df = pl.DataFrame(records)\n",
    "    \n",
    "    return passages_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd84b27d",
   "metadata": {},
   "outputs": [],
   "source": [
    "passages_df = get_passages()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5bf38ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "passages_df.write_parquet(OUTPUT_DIR + \"passages.parquet\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
