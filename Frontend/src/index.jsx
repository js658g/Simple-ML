import React from 'react';
import ReactDOM from 'react-dom';
import { Provider } from 'react-redux';
import store from './reduxStore';

import './index.css';
import App from './App';
import * as serviceWorker from './serviceWorker';

import TextEditorWrapper from './components/EditorView/TextEditor/TextEditorWrapper';
import createDefaultXtextServiceListeners from './serverConnection/XtextDefaultListeners';

import afterReactInit from './debugging/afterReactInit';
import { exposeToBrowserConsole } from './debugging/exposeToBrowserConsole';


window.loadEditor((xtextEditor) => {
    window.loadEditor = undefined;
    TextEditorWrapper.create(xtextEditor);

    createDefaultXtextServiceListeners();

    ReactDOM.render(
        <Provider store={store}>
            <App/>
        </Provider>,
        document.getElementById('root'));

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
    serviceWorker.unregister();

    afterReactInit();
    exposeToBrowserConsole();
});
