import pandas as pd
import uuid
import os 

class CSVManager:
    def __init__(self, filename: str):
        self.filename = filename

    def create_template(self) -> None:
        if os.path.exists(self.filename) == False:
            print("File Does Not Exist !")
            x = pd.DataFrame({'id' : [], 'class_type' : [], "file_name" : []})
            x.to_csv(self.filename, index = False)

    def read_csv(self) -> pd.DataFrame:
        self.create_template()
        return pd.read_csv(self.filename)

    def write_csv(self, data: pd.DataFrame):
        data.to_csv(self.filename, index=False)

    def update(self, id: str, c: str, fp: str):
        data = self.read_csv()
        if id in data['id'].values:
            data.loc[data['id'] == id, ['class_type', 'file_name']] = [c, fp]
        else:
            new_row = pd.DataFrame({'id': [id], 'class_type': [c], 'file_name': [fp]})
            data    = pd.concat([data, new_row], ignore_index=True)
        self.write_csv(data)

    def generate(self) -> str:
        data = self.read_csv()
        existing_ids = set(data['id'].values)
        while True:
            new_id = str(uuid.uuid4().hex)
            if new_id not in existing_ids:
                return new_id


if __name__ == "__main__":

    # Usage
    csv_manager = CSVManager('your_file.csv')
    csv_manager.update('123', 'A', 'file_path.jpg')
    new_id = csv_manager.generate()
    print(f'Generated new ID: {new_id}')