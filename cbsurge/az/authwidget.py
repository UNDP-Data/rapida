import jwt
import ipywidgets as widgets
from IPython.display import display
from cbsurge.util.in_notebook import in_notebook
from cbsurge.az.surgeauth import SurgeTokenCredential
out = widgets.Output(layout={'padding': '25px'})
user_name_w = widgets.Text(
    value='',
    placeholder='...enter your UNDP email address',
    description='Email:',
    disabled=False
)

password_w = widgets.Password(
    value='',
    placeholder='Enter password',
    description='Password:',
    disabled=False
)

auth_button = widgets.Button(
    description='Authenticate',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to authenticate',
    icon='user' # (FontAwesome names without the `fa-` prefix)
)




mfa_number_w = widgets.IntText(
    value=None,
    description='Any:',
    disabled=True
)

auth_widget = widgets.HBox(children=[user_name_w, password_w, auth_button], layout=widgets.Layout(border='1px solid gray', padding='15px', align_items='flex-end'))
auth_container = widgets.VBox(children=[out, auth_widget])


@out.capture(clear_output=True)
def authenticate(b):

    email, passwd = user_name_w.value, password_w.value
    if email and passwd:
        print(email, passwd)
        assert '@' in email, f'Invalid email address "{email}"'
        assert passwd not in ('', None), f'Invalid password'
        credential = SurgeTokenCredential()
        return credential.get_token(credential.STORAGE_SCOPE, auth_widget=None)



@out.capture(clear_output=True)
def decode_id_token(id_token):
    """Decodes an Azure AD ID token and extracts user details."""
    try:
        return jwt.decode(id_token, options={"verify_signature": False})
    except Exception as e:
        print("Error decoding token:", e)

auth_button.on_click(authenticate)

def load_ui():
    if in_notebook():
        credential = SurgeTokenCredential()
        token, exp_time = credential.fetch_token(credential.STORAGE_SCOPE)
        if credential.token is not None:
            info = decode_id_token(credential.token['id_token'])
            auth_button.description = f'{info["name"]}'
            user_name_w.layout.visibility = 'hidden'
            password_w.layout.visibility = 'hidden'

        display(auth_container)
        return auth_widget

