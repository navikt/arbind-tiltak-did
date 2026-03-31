select innsatsgruppe_beskrivelse, count(*) from dvh_arb_tiltak.FAK_TILTAK_UPRIORITERT_MND fak
left join dvh_arb_tiltak.dim_tiltakstype dim
on fak.fk_dim_tiltakstype = dim.pk_dim_tiltakstype
left join dvh_arb_kartlegging.DIM_INNSATSGRUPPE_14A dim14
on dim14.pk_dim_innsatsgruppe_14a = fak.fk_dim_innsatsgruppe_14a
where tiltaksnavn = 'Midlertidig lønnstilskudd'
and substr(periode,1, 6) >= 202010	
and substr(periode,1, 6) <= 202509
and prioritert = 1
group by innsatsgruppe_beskrivelse
order by innsatsgruppe_beskrivelse
; --464134