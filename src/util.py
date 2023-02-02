import dropbox

with open('token.txt', 'r') as f:
    token = f.read()

def to_dropbox(dataframe, path):
    dbx = dropbox.Dropbox(token)

    df_string = dataframe.to_csv(index=False)
    db_bytes = bytes(df_string, 'utf8')
    
    dbx.files_upload(
        f=db_bytes,
        path=path,
        mode=dropbox.files.WriteMode.overwrite
    )

def get_url_of_file(path):
    dbx = dropbox.Dropbox(token)
    url = dbx.sharing_create_shared_link(f'/{path}').url
    return url