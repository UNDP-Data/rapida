import datetime
import ipywidgets as widgets
from oauthlib.oauth2 import OAuth2Error
import asyncio
import jwt
from cbsurge.util.in_notebook import in_notebook
from cbsurge.az.surgeauth import SurgeTokenCredential
from IPython.display import display
import logging
from cbsurge.util.setup_logger import setup_logger


if in_notebook():
    logger = setup_logger(make_root=True)
else:
    logger = logging.getLogger(__name__)





class AuthWidget:
    _instance = None
    _rendered = False
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.credential = SurgeTokenCredential()

        # Consistent layout and label alignment
        text_input_layout = widgets.Layout(width='200px')
        input_style = {'description_width': '100px'}

        self.email_w = widgets.Text(
            value='ioan.ferencik@undp.org',
            placeholder='...enter your UNDP email address',
            description='Email:',
            layout=text_input_layout,

        )

        self.password_w = widgets.Password(
            placeholder='Enter password',
            description='Password:',
            layout=text_input_layout,

        )

        self.auth_button = widgets.Button(
            description='Authenticate',
            button_style='info',
            icon='user',
            layout=widgets.Layout(width='150px', align_self='flex-end')
        )

        self.feedback_html = widgets.HTML(value="",
                                     layout={'border': '0px solid grey', 'background': 'silver', 'padding': '5px'})

        self.auth_widget = widgets.HBox(children=[self.email_w, self.password_w, self.auth_button],
                                   layout=widgets.Layout(padding='2px', justify_content='flex-end',
                                                         align_items='center'))
        self.container = widgets.VBox(
            children=[self.auth_widget, self.feedback_html],
            layout=widgets.Layout(
                align_items='flex-end',  # Aligns the children (feedback_html and auth_widget) to the right
                justify_content='flex-start',  # Aligns the children vertically at the top
                width='100%'  # Ensures the container stretches across the available space
            )
        )
        self.auth_button.on_click(self.on_click)


    def display(self):
        if not self.__class__._rendered:
            # Display the widget container
            display(self.container)
            self.__class__._rendered = True


    def render(self):
        self.display()
        self.handle()

    def handle(self):
        
        if self.credential.authenticated:
            self._handle_authenticated()
        else:
            self._prepare_to_authenticate()
            if self.credential.token:
                asyncio.ensure_future(self.credential.get_token_async())
            if self.credential.authenticated:
                self._handle_authenticated()


    def _decode_token(self, token):
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception as e:
            self.feedback_html.value = f"<b style='color:red'>Token decode error: {e}</b>"
            return {}

    def _handle_authenticated(self):
        info = self._decode_token(self.credential.token['id_token'])
        self.email_w.layout.display = 'none'
        self.password_w.layout.display = 'none'
        self.auth_button.description = f"{info.get('name', 'Logged in')}"
        self.auth_button.button_style = 'success'
        self.feedback_html.value = ""

    def _prepare_to_authenticate(self):
        self.email_w.layout.display = 'block'
        self.password_w.layout.display = 'block'
        self.auth_button.description = f"Authenticate"
        self.auth_button.button_style = 'info'
        self.feedback_html.value = ""

    async def authenticate(self):
        email = self.email_w.value.strip()
        passwd = self.password_w.value.strip()
        self.auth_button.description = "Authenticating..."
        self.auth_button.disabled = True

        if not email or not passwd:
            self.feedback_html.value = "<b style='color:red'>Please enter both email and password.</b>"
            self.auth_button.description = "Authenticate"
            self.auth_button.disabled = False
            return

        if '@' not in email:
            self.feedback_html.value = f"<b style='color:brown'>Invalid email {email}</b>"
            self.auth_button.description = "Authenticate"
            self.auth_button.disabled = False
            return

        try:
            await self.credential.get_token_async(
            self.credential.STORAGE_SCOPE,
                email=email,
                password=passwd,
                mfa_widget=self.feedback_html
            )
            self._handle_authenticated()

        except (OAuth2Error, Exception) as e:
            self.feedback_html.value = f"<b style='color:red'>Authentication failed: {e}</b>"
            self.auth_button.button_style = 'info'
            self.auth_button.description = "Authenticate"

        finally:
            self.auth_button.disabled = False


    def on_click(self, btn):
        if self.credential.authenticated:
            local_tz = datetime.datetime.now().astimezone().tzinfo
            dt = datetime.datetime.fromtimestamp(self.credential.token['expires_at'], tz=local_tz)
            self.feedback_html.value = (f"<b style='color:brown'>You are authenticated until "
                                        f"{dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")}</b>")
            return
        asyncio.ensure_future(self.authenticate())

