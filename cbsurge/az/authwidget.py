
import logging
import ipywidgets as widgets
import asyncio
import jwt
from cbsurge.az.surgeauth import SurgeTokenCredential
from IPython.display import display



logger = logging.getLogger(__name__)



class AuthWidget:
    def __init__(self):
        self.credential = SurgeTokenCredential()

        # Widgets
        self.email_w = widgets.Text(
            placeholder='...enter your UNDP email address',
            description='Email:',
            layout=widgets.Layout(width='250px')
        )

        self.password_w = widgets.Password(
            placeholder='Enter password',
            description='Password:',
            layout=widgets.Layout(width='250px')
        )

        self.auth_button = widgets.Button(
            description='Authenticate',
            button_style='info',
            icon='user'
        )

        self.feedback_html = widgets.HTML(
            value="", layout={'border': '0px', 'background': 'whitesmoke', 'padding': '5px'}
        )

        # Layout
        self.auth_widget = widgets.HBox(
            children=[self.email_w, self.password_w, self.auth_button],
            layout=widgets.Layout(justify_content='flex-end', align_items='center', padding='4px')
        )

        self.container = widgets.VBox(
            children=[self.auth_widget, self.feedback_html],
            layout=widgets.Layout(width='100%', align_items='flex-end', justify_content='flex-start')
        )

        self.auth_button.on_click(self.on_click)

    def render(self):
        display(self.container)
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
        except Exception as e:
            self.feedback_html.value = f"<b style='color:red'>Authentication failed: {e}</b>"
        finally:
            self.auth_button.description = "Authenticate"
            self.auth_button.disabled = False

    def on_click(self, btn):
        asyncio.ensure_future(self.authenticate())
