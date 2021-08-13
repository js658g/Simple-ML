/*******************************************************************************
 * Copyright (c) 2015 itemis AG (http://www.itemis.eu) and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *******************************************************************************/

define(['xtext/services/XtextService', 'jquery'], function(XtextService, jQuery) {

    /**
     * Service class for loading resources. The resulting text is passed to the editor context.
     */
    function GetProcessProposalsService(serviceUrl, resourceId) {
        this.initialize(serviceUrl, 'getProcessProposals', resourceId);
    };

    GetProcessProposalsService.prototype = new XtextService();

    GetProcessProposalsService.prototype._initServerData = function(serverData, editorContext, params) {
        return {
            suppressContent: true,
            httpMethod: 'GET'
        };
    };

    GetProcessProposalsService.prototype._getSuccessCallback = function(editorContext, params, deferred) {
        return function(result) {
            deferred.resolve(result);
        }
    };

    return GetProcessProposalsService;
});