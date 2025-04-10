import ipywidgets
import jwt
import ipywidgets as widgets
from IPython.display import display
from cbsurge.util.in_notebook import in_notebook
from cbsurge.az.surgeauth import SurgeTokenCredential
import logging
import asyncio
logger = logging.getLogger(__name__)

#out = widgets.Output(layout={'padding': '25px'})

user_name_w = widgets.Text(
    value='',
    placeholder='...enter your UNDP email address',
    description='Email:',
    disabled=False,
    layout=widgets.Layout(visibility='visible')
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

feedback_html = widgets.HTML(value="", layout={'border': '0px solid grey', 'background':'silver', 'padding':'5px'})

auth_widget = widgets.HBox(children=[user_name_w, password_w, auth_button], layout=widgets.Layout(padding='2px', justify_content='flex-end', align_items='center'))
auth_container = widgets.VBox(
    children=[auth_widget,feedback_html],
    layout=widgets.Layout(
        align_items='flex-end',  # Aligns the children (feedback_html and auth_widget) to the right
        justify_content='flex-start',  # Aligns the children vertically at the top
        width='100%'  # Ensures the container stretches across the available space
    )
)



#@out.capture(clear_output=True)
async def authenticate():
    email, passwd = user_name_w.value, password_w.value
    if email and passwd:
        assert '@' in email, f'Invalid email address "{email}"'
        assert passwd not in ('', None), f'Invalid password'
        credential = SurgeTokenCredential()
        res = await credential.get_token_async(credential.STORAGE_SCOPE, email=email, password=passwd, mfa_widget=feedback_html)
        feedback_html.value=''
        info = decode_id_token(credential.token['id_token'])
        auth_button.description = f'{info["name"]}'
        # auth_button.layout.visibility = 'hidden'
        user_name_w.layout = widgets.Layout(display='none')
        password_w.layout = widgets.Layout(display='none')

def on_click(b):
    asyncio.ensure_future(authenticate())

#@out.capture(clear_output=True)
def decode_id_token(id_token):
    """Decodes an Azure AD ID token and extracts user details."""
    try:
        return jwt.decode(id_token, options={"verify_signature": False})
    except Exception as e:
        print("Error decoding token:", e)

auth_button.on_click(on_click)

def load_ui():
    credential = SurgeTokenCredential()
    display(auth_container)
    if credential.authenticated:
        info = decode_id_token(credential.token['id_token'])
        auth_button.description = f'{info["name"]}'
        #auth_button.layout.visibility = 'hidden'
        user_name_w.layout = widgets.Layout(display='none')
        password_w.layout = widgets.Layout(display='none')
        #feedback_html.value = f'<b>UNDP user:  {info["name"]}</b> '

    else:
        if auth_button.disabled:
            auth_button.disabled = False
        if auth_button.description != 'Authenticate':
            auth_button.description = 'Authenticate'



