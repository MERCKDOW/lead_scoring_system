-- join table expanded-nebula-754.sanbox_cdow.golden_records and  
-- expanded-nebula-754.SFDC.lead on std_email_address and Email and select only the filed in expanded-nebula-754.sanbox_cdow.golden_records std_email_address
SELECT
  t1.std_email_address,
  t2.Is_Won

FROM
  `expanded-nebula-754`.`sanbox_cdow`.`golden_records` AS t1
  
INNER JOIN
  `expanded-nebula-754`.`sanbox_cdow`.`Leads_Won_Email` AS t2
ON
  t1.std_email_address = t2.Email

#WHERE t2.Is_Won = TRUE;  