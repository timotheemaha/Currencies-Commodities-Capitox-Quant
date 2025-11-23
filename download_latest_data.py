import gdown

commodities_url = "https://drive.google.com/drive/folders/1jFZvTNlQnoFRB3WVw4cKewKW15olW95U"
macro_url = "https://drive.google.com/drive/folders/1XdGraH2g42_K-SIHXbc7k98JBiPz6jyz"

gdown.download_folder(commodities_url, quiet=False, use_cookies=False)
gdown.download_folder(macro_url, quiet=False, use_cookies=False)
