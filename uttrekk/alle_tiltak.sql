select *
from (
    select ned.aarmnd, ned.nav_region_navn, count(*) as cnt
    from dvh_arb_tiltak.FAK_TILTAK_UPRIORITERT_MND tiltak
    inner join nedbrytning ned
        on ned.fk_person1 = tiltak.fk_person1
        and ned.aarmnd = substr(tiltak.periode, 1, 6)
    group by ned.aarmnd, ned.nav_region_navn
)
pivot (
    sum(cnt)
    for nav_region_navn in (
        --'Nasjonal oppfølgingsenhet',
        'Nav Agder',
        'Nav Innlandet',
        'Nav Møre og Romsdal',
        'Nav Nordland',
        'Nav Oslo',
        'Nav Rogaland',
        'Nav Troms og Finnmark',
        'Nav Trøndelag',
        'Nav Vestfold og Telemark',
        'Nav Vestland',
        'Nav Vest-Viken',
        'Nav Øst-Viken'
    )
)
order by aarmnd
;


select *
from (
    select ned.aarmnd_dato , ned.nav_region_navn, count(*) as cnt
    from dvh_arb_tiltak.FAK_TILTAK_UPRIORITERT_MND tiltak
    inner join nedbrytning ned
        on ned.fk_person1 = tiltak.fk_person1
        and ned.aarmnd = substr(tiltak.periode, 1, 6)
    left join dvh_arb_tiltak.dim_tiltakstype dim
    on tiltak.fk_dim_tiltakstype = dim.pk_dim_tiltakstype
    where tiltaksnavn = 'Midlertidig lønnstilskudd'
    group by ned.aarmnd_dato, ned.nav_region_navn
)
pivot (
    sum(cnt)
    for nav_region_navn in (
        --'Nasjonal oppfølgingsenhet',
        'Nav Agder',
        'Nav Innlandet',
        'Nav Møre og Romsdal',
        'Nav Nordland',
        'Nav Oslo',
        'Nav Rogaland',
        'Nav Troms og Finnmark',
        'Nav Trøndelag',
        'Nav Vestfold og Telemark',
        'Nav Vestland',
        'Nav Vest-Viken',
        'Nav Øst-Viken'
    )
)
order by aarmnd_dato
;

select ned.aarmnd_dato , ned.nav_enhet_navn, count(*) as cnt
from dvh_arb_tiltak.FAK_TILTAK_UPRIORITERT_MND tiltak
inner join nedbrytning ned
    on ned.fk_person1 = tiltak.fk_person1
    and ned.aarmnd = substr(tiltak.periode, 1, 6)
left join dvh_arb_tiltak.dim_tiltakstype dim
on tiltak.fk_dim_tiltakstype = dim.pk_dim_tiltakstype
where tiltaksnavn = 'Midlertidig lønnstilskudd'
group by ned.aarmnd_dato, ned.nav_enhet_navn
;
