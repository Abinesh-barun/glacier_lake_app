"""Extra page for glacial_lake_app.py: alert contacts."""
import streamlit as st

import contacts_db as db


def _hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def page_area_contacts():
    _hero("Alert Contacts",
          "Authorities and residents who receive region risk alerts. Only add people who agreed to it.")

    res = st.session_state.get("region_result")
    names = sorted({p["place"] for pl in res["places"].values() for p in pl}) if res else []

    with st.container(border=True):
        st.markdown("**Add a contact**")
        with st.form("area_add_contact", clear_on_submit=True):
            n = st.text_input("Name")
            e = st.text_input("Email")
            role = st.selectbox("Role", ["resident", "authority"])
            place = st.text_input("Place (residents: the village name; authorities: blank = all areas)")
            submitted = st.form_submit_button("Add contact")
        if names:
            st.caption("Place names from the last analysis: " + ", ".join(names))
        if submitted:
            try:
                db.add_contact(n, e, role, place)
                st.success("Contact added.")
            except ValueError as err:
                st.error(str(err))

    contacts = db.list_contacts()
    st.subheader(f"All contacts ({len(contacts)})")
    if not contacts:
        st.info("No contacts yet.")
    for c in contacts:
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([3, 4, 2, 2, 1.6], vertical_alignment="center")
            c1.write(f"**{c['name']}**")
            c2.write(c["email"])
            c3.write(c["role"])
            c4.write(c["place"] or "all areas")
            if c5.button("Remove", key=f"area_del_{c['id']}", use_container_width=True):
                db.delete_contact(c["id"])
                st.rerun()
