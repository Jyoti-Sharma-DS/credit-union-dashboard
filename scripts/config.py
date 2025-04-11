# config.py

task_type_mapping = {
        # Inbound Calls
        'Inbound Call': 'Inbound Calls',
        'Inb Followup Call': 'Inbound Calls',
        'Inb Audit Followup': 'Inbound Calls',
        'Inb Send Document Error Notice': 'Inbound Calls',

        # Outbound Calls
        'Call Employer': 'Outbound Calls',
        'Followup Call': 'Outbound Calls',
        'Call SE': 'Outbound Calls',
        'SE Followup Call': 'Outbound Calls',
        'Reverify Followup Call': 'Outbound Calls',
        'Rever Audit Followup': 'Outbound Calls',

        # Research
        'Research': 'Research',
        'Research Alt Number': 'Research',
        'Research Followup': 'Research',
        'Research Helpdesk': 'Research',
        'Research 2nd': 'Research',
        'SE Research': 'Research',
        'SE Research Followup': 'Research',

        # Consent/Documents
        'Review Consent Form': 'Consent/Documents',
        'Request Consent Form': 'Consent/Documents',
        'Send Document Error Notice': 'Consent/Documents',
        'Transcribe Request': 'Consent/Documents',
        'Resume Request': 'Consent/Documents',

        # Employee Input
        'External Employee Input': 'Employee Input',
        'Employee Input': 'Employee Input',

        # Audit & Verification
        'External VOE': 'Audit & Verification',
        'Audit Followup': 'Audit & Verification',
        'Reverify With Employer': 'Audit & Verification',
    }

outcome_mapping = {
    
    # Successful Completion
    'APPROVED': 'Successful Completion',
    'Match': 'Successful Completion',
    'EMPLOYEEINPUT': 'Successful Completion',
    'EMPLOYEE INPUT UPDATED': 'Successful Completion',
    'RESEARCH COMPLETE': 'Successful Completion',
    'Resume Complete': 'Successful Completion',
    'Resume Complete - No Corrections': 'Successful Completion',
    'VOE complete via External 3rd Party': 'Successful Completion',
    'VOEI complete via External 3rd Party': 'Successful Completion',
    'SE order completed': 'Successful Completion',
    'VOEI order complete': 'Successful Completion',
    'VOE written completed': 'Successful Completion',
    'VOI written completed': 'Successful Completion',
    'Verbal VOE completed': 'Successful Completion',
    'SE - Completed with CPA': 'Successful Completion',
    'Existing Work Number record': 'Successful Completion',
    'VOE Complete - Audit Follow Up': 'Successful Completion',
    'VOI Complete - Audit Follow Up': 'Successful Completion',
    
    # Contact Issues
    'Ring No Answer': 'Contact Issues',
    'Left Voice Message': 'Contact Issues',
    'Busy Signal': 'Contact Issues',
    'Number Disconnected': 'Contact Issues',
    'Wrong Number': 'Contact Issues',
    'No Valid Number Found': 'Contact Issues',
    "Borrower's Direct Line": "Contact Issues",
    "Borrower’s Direct Line": "Contact Issues",
    
    # Employer/Employee Issues
    'Employer contacted - no such employee': 'Employer/Employee Issues',
    'Employer unable to provide requested information': 'Employer/Employee Issues',
    'Employer would not provide requested information': 'Employer/Employee Issues',
    'Employer refused duplicate request': 'Employer/Employee Issues',
    'Employer will only mail verification': 'Employer/Employee Issues',
    'Employer has not returned calls': 'Employer/Employee Issues',
    'Employer contacted - Employer/Employee Are Related': 'Employer/Employee Issues',
    'Employer contacted - Self employed applicant': 'Employer/Employee Issues',
    'Unable to provide verification (miscellaneous)': 'Employer/Employee Issues',
    'Wrong Name': 'Employer/Employee Issues',
    'Wrong SSN on submission': 'Employer/Employee Issues',
    'EXTERNAL': 'External/Third-party Interaction',
    
    # External/Third-party Interaction
    'EXTERNAL CONSENT REQUIRED': 'External/Third-party Interaction',
    'EXTERNAL EMPLOYEE INPUT': 'External/Third-party Interaction',
    'CONSENT REQUESTED': 'External/Third-party Interaction',
    'Authorization Requested': 'External/Third-party Interaction',
    
    # Task Management Actions
    'Requeue Task': 'Task Management Actions',
    'RESUBMIT': 'Task Management Actions',
    'RETURNTOCALL': 'Task Management Actions',
    'SEND DOCUMENTS': 'Task Management Actions',
    'Submit': 'Task Management Actions',
    'CANCEL - 43002': 'Task Management Actions',
    'Client requested cancellation within 2 hours': 'Task Management Actions',
    'Client requested cancellation within 2 days': 'Task Management Actions',
    'Client requested cancellation after 2 days': 'Task Management Actions',
    'AUDITREQUIRED': 'Task Management Actions',
    'BYPASSCALL': 'Task Management Actions',
    'BYPASSPRIORITYCALL': 'Task Management Actions',
    'Fax': 'Task Management Actions',
    'Email': 'Task Management Actions',
    'rescheduled': 'Task Management Actions',

    # Rejected / No Match
    'REJECTED': 'Rejected / No Match',
    'No Match': 'Rejected / No Match'
    }


# Cluster mapping 
clusters_mapping = {
    'Cluster' : [ 2,0,1],
    ' Agent Characteristics' : [ 'High Productivity, Efficiency, High Success',
                         'Moderate performance, balanced metrics',
                         'Lower productivity, high variability, low success'
                        ],
    'Recommendations for future Growth' : ['Recorgnized and leverage as mentors',
                         'Regular monitoring, moderate coaching',
                         'Immediate coaching, training, task reassignment'
                        ]
}

cluster_dict = {
    2: "High Productivity, High Efficiency, High Success",
    0: "Moderate Performance, Balanced Metrics",
    1:"Lower Productivity, High Variability, Low Success"
  
}

cluster_rec = {
    1:  "Immediate coaching, training, task reassignment",
    2: "Recognize and leverage as mentors",
    0 : "Regular monitoring, moderate coaching"
  }